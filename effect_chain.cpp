#include <epoxy/gl.h>
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <set>
#include <stack>
#include <utility>
#include <vector>
#include <Eigen/Core>

#include "alpha_division_effect.h"
#include "alpha_multiplication_effect.h"
#include "colorspace_conversion_effect.h"
#include "dither_effect.h"
#include "effect.h"
#include "effect_chain.h"
#include "effect_util.h"
#include "gamma_compression_effect.h"
#include "gamma_expansion_effect.h"
#include "init.h"
#include "input.h"
#include "resource_pool.h"
#include "util.h"
#include "ycbcr_conversion_effect.h"

using namespace Eigen;
using namespace std;

namespace movit {

EffectChain::EffectChain(float aspect_nom, float aspect_denom, ResourcePool *resource_pool, GLenum intermediate_format)
	: aspect_nom(aspect_nom),
	  aspect_denom(aspect_denom),
	  output_color_rgba(false),
	  output_color_ycbcr(false),
	  dither_effect(NULL),
	  intermediate_format(intermediate_format),
	  num_dither_bits(0),
	  output_origin(OUTPUT_ORIGIN_BOTTOM_LEFT),
	  finalized(false),
	  resource_pool(resource_pool),
	  do_phase_timing(false) {
	if (resource_pool == NULL) {
		this->resource_pool = new ResourcePool();
		owns_resource_pool = true;
	} else {
		owns_resource_pool = false;
	}

	// Generate a VBO with some data in (shared position and texture coordinate data).
	float vertices[] = {
		0.0f, 2.0f,
		0.0f, 0.0f,
		2.0f, 0.0f
	};
	vbo = generate_vbo(2, GL_FLOAT, sizeof(vertices), vertices);
}

EffectChain::~EffectChain()
{
	for (unsigned i = 0; i < nodes.size(); ++i) {
		delete nodes[i]->effect;
		delete nodes[i];
	}
	for (unsigned i = 0; i < phases.size(); ++i) {
		resource_pool->release_glsl_program(phases[i]->glsl_program_num);
		delete phases[i];
	}
	if (owns_resource_pool) {
		delete resource_pool;
	}
	glDeleteBuffers(1, &vbo);
	check_error();
}

Input *EffectChain::add_input(Input *input)
{
	assert(!finalized);
	inputs.push_back(input);
	add_node(input);
	return input;
}

void EffectChain::add_output(const ImageFormat &format, OutputAlphaFormat alpha_format)
{
	assert(!finalized);
	assert(!output_color_rgba);
	output_format = format;
	output_alpha_format = alpha_format;
	output_color_rgba = true;
}

void EffectChain::add_ycbcr_output(const ImageFormat &format, OutputAlphaFormat alpha_format,
                                   const YCbCrFormat &ycbcr_format, YCbCrOutputSplitting output_splitting)
{
	assert(!finalized);
	assert(!output_color_ycbcr);
	output_format = format;
	output_alpha_format = alpha_format;
	output_color_ycbcr = true;
	output_ycbcr_format = ycbcr_format;
	output_ycbcr_splitting = output_splitting;

	assert(ycbcr_format.chroma_subsampling_x == 1);
	assert(ycbcr_format.chroma_subsampling_y == 1);
}

Node *EffectChain::add_node(Effect *effect)
{
	for (unsigned i = 0; i < nodes.size(); ++i) {
		assert(nodes[i]->effect != effect);
	}

	Node *node = new Node;
	node->effect = effect;
	node->disabled = false;
	node->output_color_space = COLORSPACE_INVALID;
	node->output_gamma_curve = GAMMA_INVALID;
	node->output_alpha_type = ALPHA_INVALID;
	node->needs_mipmaps = false;
	node->one_to_one_sampling = false;

	nodes.push_back(node);
	node_map[effect] = node;
	effect->inform_added(this);
	return node;
}

void EffectChain::connect_nodes(Node *sender, Node *receiver)
{
	sender->outgoing_links.push_back(receiver);
	receiver->incoming_links.push_back(sender);
}

void EffectChain::replace_receiver(Node *old_receiver, Node *new_receiver)
{
	new_receiver->incoming_links = old_receiver->incoming_links;
	old_receiver->incoming_links.clear();
	
	for (unsigned i = 0; i < new_receiver->incoming_links.size(); ++i) {
		Node *sender = new_receiver->incoming_links[i];
		for (unsigned j = 0; j < sender->outgoing_links.size(); ++j) {
			if (sender->outgoing_links[j] == old_receiver) {
				sender->outgoing_links[j] = new_receiver;
			}
		}
	}	
}

void EffectChain::replace_sender(Node *old_sender, Node *new_sender)
{
	new_sender->outgoing_links = old_sender->outgoing_links;
	old_sender->outgoing_links.clear();
	
	for (unsigned i = 0; i < new_sender->outgoing_links.size(); ++i) {
		Node *receiver = new_sender->outgoing_links[i];
		for (unsigned j = 0; j < receiver->incoming_links.size(); ++j) {
			if (receiver->incoming_links[j] == old_sender) {
				receiver->incoming_links[j] = new_sender;
			}
		}
	}	
}

void EffectChain::insert_node_between(Node *sender, Node *middle, Node *receiver)
{
	for (unsigned i = 0; i < sender->outgoing_links.size(); ++i) {
		if (sender->outgoing_links[i] == receiver) {
			sender->outgoing_links[i] = middle;
			middle->incoming_links.push_back(sender);
		}
	}
	for (unsigned i = 0; i < receiver->incoming_links.size(); ++i) {
		if (receiver->incoming_links[i] == sender) {
			receiver->incoming_links[i] = middle;
			middle->outgoing_links.push_back(receiver);
		}
	}

	assert(middle->incoming_links.size() == middle->effect->num_inputs());
}

GLenum EffectChain::get_input_sampler(Node *node, unsigned input_num) const
{
	assert(node->effect->needs_texture_bounce());
	assert(input_num < node->incoming_links.size());
	assert(node->incoming_links[input_num]->bound_sampler_num >= 0);
	assert(node->incoming_links[input_num]->bound_sampler_num < 8);
	return GL_TEXTURE0 + node->incoming_links[input_num]->bound_sampler_num;
}

GLenum EffectChain::has_input_sampler(Node *node, unsigned input_num) const
{
	assert(input_num < node->incoming_links.size());
	return node->incoming_links[input_num]->bound_sampler_num >= 0 &&
		node->incoming_links[input_num]->bound_sampler_num < 8;
}

void EffectChain::find_all_nonlinear_inputs(Node *node, vector<Node *> *nonlinear_inputs)
{
	if (node->output_gamma_curve == GAMMA_LINEAR &&
	    node->effect->effect_type_id() != "GammaCompressionEffect") {
		return;
	}
	if (node->effect->num_inputs() == 0) {
		nonlinear_inputs->push_back(node);
	} else {
		assert(node->effect->num_inputs() == node->incoming_links.size());
		for (unsigned i = 0; i < node->incoming_links.size(); ++i) {
			find_all_nonlinear_inputs(node->incoming_links[i], nonlinear_inputs);
		}
	}
}

Effect *EffectChain::add_effect(Effect *effect, const vector<Effect *> &inputs)
{
	assert(!finalized);
	assert(inputs.size() == effect->num_inputs());
	Node *node = add_node(effect);
	for (unsigned i = 0; i < inputs.size(); ++i) {
		assert(node_map.count(inputs[i]) != 0);
		connect_nodes(node_map[inputs[i]], node);
	}
	return effect;
}

// ESSL doesn't support token pasting. Replace PREFIX(x) with <effect_id>_x.
string replace_prefix(const string &text, const string &prefix)
{
	string output;
	size_t start = 0;

	while (start < text.size()) {
		size_t pos = text.find("PREFIX(", start);
		if (pos == string::npos) {
			output.append(text.substr(start, string::npos));
			break;
		}

		output.append(text.substr(start, pos - start));
		output.append(prefix);
		output.append("_");

		pos += strlen("PREFIX(");
	
		// Output stuff until we find the matching ), which we then eat.
		int depth = 1;
		size_t end_arg_pos = pos;
		while (end_arg_pos < text.size()) {
			if (text[end_arg_pos] == '(') {
				++depth;
			} else if (text[end_arg_pos] == ')') {
				--depth;
				if (depth == 0) {
					break;
				}
			}
			++end_arg_pos;
		}
		output.append(text.substr(pos, end_arg_pos - pos));
		++end_arg_pos;
		assert(depth == 0);
		start = end_arg_pos;
	}
	return output;
}

namespace {

template<class T>
void extract_uniform_declarations(const vector<Uniform<T> > &effect_uniforms,
                                  const string &type_specifier,
                                  const string &effect_id,
                                  bool in_ubo_block,
                                  vector<Uniform<T> > *phase_uniforms,
                                  string *glsl_string)
{
	for (unsigned i = 0; i < effect_uniforms.size(); ++i) {
		phase_uniforms->push_back(effect_uniforms[i]);
		phase_uniforms->back().prefix = effect_id;

		if (!in_ubo_block) {
			*glsl_string += "uniform ";
		}
		*glsl_string += type_specifier + " " + effect_id + "_" + effect_uniforms[i].name + ";\n";
	}
}

template<class T>
void extract_uniform_array_declarations(const vector<Uniform<T> > &effect_uniforms,
                                        const string &type_specifier,
                                        const string &effect_id,
                                        bool in_ubo_block,
                                        vector<Uniform<T> > *phase_uniforms,
                                        string *glsl_string)
{
	for (unsigned i = 0; i < effect_uniforms.size(); ++i) {
		phase_uniforms->push_back(effect_uniforms[i]);
		phase_uniforms->back().prefix = effect_id;

		if (!in_ubo_block) {
			*glsl_string += "uniform ";
		}

		char buf[256];
		snprintf(buf, sizeof(buf), "%s %s_%s[%d];\n",
			type_specifier.c_str(), effect_id.c_str(),
			effect_uniforms[i].name.c_str(),
			int(effect_uniforms[i].num_values));
		*glsl_string += buf;
	}
}

template<class T>
void collect_uniform_locations(GLuint glsl_program_num, vector<Uniform<T> > *phase_uniforms)
{
	for (unsigned i = 0; i < phase_uniforms->size(); ++i) {
		Uniform<T> &uniform = (*phase_uniforms)[i];
		uniform.location = get_uniform_location(glsl_program_num, uniform.prefix, uniform.name);
		get_uniform_offset_and_size(glsl_program_num, uniform.prefix, uniform.name, &uniform.ubo_offset, &uniform.ubo_num_elem);
	}
}

}  // namespace

void EffectChain::compile_glsl_program(Phase *phase)
{
	string frag_shader_header = read_version_dependent_file("header", "frag");
	string frag_shader = "";

	// Create functions and uniforms for all the texture inputs that we need.
	for (unsigned i = 0; i < phase->inputs.size(); ++i) {
		Node *input = phase->inputs[i]->output_node;
		char effect_id[256];
		sprintf(effect_id, "in%u", i);
		phase->effect_ids.insert(make_pair(input, effect_id));
	
		frag_shader += string("uniform sampler2D tex_") + effect_id + ";\n";
		frag_shader += string("vec4 ") + effect_id + "(vec2 tc) {\n";
		frag_shader += "\treturn tex2D(tex_" + string(effect_id) + ", tc);\n";
		frag_shader += "}\n";
		frag_shader += "\n";

		Uniform<int> uniform;
		uniform.name = effect_id;
		uniform.value = &phase->input_samplers[i];
		uniform.prefix = "tex";
		uniform.num_values = 1;
		uniform.location = -1;
		phase->uniforms_sampler2d.push_back(uniform);
	}

	// Give each effect in the phase its own ID.
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		char effect_id[256];
		sprintf(effect_id, "eff%u", i);
		phase->effect_ids.insert(make_pair(node, effect_id));
	}

	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		const string effect_id = phase->effect_ids[node];
		if (node->incoming_links.size() == 1) {
			frag_shader += string("#define INPUT ") + phase->effect_ids[node->incoming_links[0]] + "\n";
		} else {
			for (unsigned j = 0; j < node->incoming_links.size(); ++j) {
				char buf[256];
				sprintf(buf, "#define INPUT%d %s\n", j + 1, phase->effect_ids[node->incoming_links[j]].c_str());
				frag_shader += buf;
			}
		}
	
		frag_shader += "\n";
		frag_shader += string("#define FUNCNAME ") + effect_id + "\n";
		frag_shader += replace_prefix(node->effect->output_fragment_shader(), effect_id);
		frag_shader += "#undef PREFIX\n";
		frag_shader += "#undef FUNCNAME\n";
		if (node->incoming_links.size() == 1) {
			frag_shader += "#undef INPUT\n";
		} else {
			for (unsigned j = 0; j < node->incoming_links.size(); ++j) {
				char buf[256];
				sprintf(buf, "#undef INPUT%d\n", j + 1);
				frag_shader += buf;
			}
		}
		frag_shader += "\n";
	}
	frag_shader += string("#define INPUT ") + phase->effect_ids[phase->effects.back()] + "\n";

	// If we're the last phase, add the right #defines for Y'CbCr multi-output as needed.
	vector<string> frag_shader_outputs;  // In order.
	if (phase->output_node->outgoing_links.empty() && output_color_ycbcr) {
		switch (output_ycbcr_splitting) {
		case YCBCR_OUTPUT_INTERLEAVED:
			// No #defines set.
			frag_shader_outputs.push_back("FragColor");
			break;
		case YCBCR_OUTPUT_SPLIT_Y_AND_CBCR:
			frag_shader += "#define YCBCR_OUTPUT_SPLIT_Y_AND_CBCR 1\n";
			frag_shader_outputs.push_back("Y");
			frag_shader_outputs.push_back("Chroma");
			break;
		case YCBCR_OUTPUT_PLANAR:
			frag_shader += "#define YCBCR_OUTPUT_PLANAR 1\n";
			frag_shader_outputs.push_back("Y");
			frag_shader_outputs.push_back("Cb");
			frag_shader_outputs.push_back("Cr");
			break;
		default:
			assert(false);
		}

		if (output_color_rgba) {
			// Note: Needs to come in the header, because not only the
			// output needs to see it (YCbCrConversionEffect and DitherEffect
			// do, too).
			frag_shader_header += "#define YCBCR_ALSO_OUTPUT_RGBA 1\n";
			frag_shader_outputs.push_back("RGBA");
		}
	}
	frag_shader.append(read_file("footer.frag"));

	// Collect uniforms from all effects and output them. Note that this needs
	// to happen after output_fragment_shader(), even though the uniforms come
	// before in the output source, since output_fragment_shader() is allowed
	// to register new uniforms (e.g. arrays that are of unknown length until
	// finalization time).
	string frag_shader_uniforms = "";
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		const bool in_ubo_block = true;  // TODO: Check for the extension.
		Node *node = phase->effects[i];
		Effect *effect = node->effect;
		const string effect_id = phase->effect_ids[node];
		extract_uniform_declarations(effect->uniforms_bool, "bool", effect_id, in_ubo_block, &phase->uniforms_bool, &frag_shader_uniforms);
		extract_uniform_declarations(effect->uniforms_int, "int", effect_id, in_ubo_block, &phase->uniforms_int, &frag_shader_uniforms);
		extract_uniform_declarations(effect->uniforms_float, "float", effect_id, in_ubo_block, &phase->uniforms_float, &frag_shader_uniforms);
		extract_uniform_declarations(effect->uniforms_vec2, "vec2", effect_id, in_ubo_block, &phase->uniforms_vec2, &frag_shader_uniforms);
		extract_uniform_declarations(effect->uniforms_vec3, "vec3", effect_id, in_ubo_block, &phase->uniforms_vec3, &frag_shader_uniforms);
		extract_uniform_declarations(effect->uniforms_vec4, "vec4", effect_id, in_ubo_block, &phase->uniforms_vec4, &frag_shader_uniforms);
		extract_uniform_array_declarations(effect->uniforms_float_array, "float", effect_id, in_ubo_block, &phase->uniforms_float, &frag_shader_uniforms);
		extract_uniform_array_declarations(effect->uniforms_vec2_array, "vec2", effect_id, in_ubo_block, &phase->uniforms_vec2, &frag_shader_uniforms);
		extract_uniform_array_declarations(effect->uniforms_vec3_array, "vec3", effect_id, in_ubo_block, &phase->uniforms_vec3, &frag_shader_uniforms);
		extract_uniform_array_declarations(effect->uniforms_vec4_array, "vec4", effect_id, in_ubo_block, &phase->uniforms_vec4, &frag_shader_uniforms);
		extract_uniform_declarations(effect->uniforms_mat3, "mat3", effect_id, in_ubo_block, &phase->uniforms_mat3, &frag_shader_uniforms);
	}
	if (!frag_shader_uniforms.empty()) {
		frag_shader_uniforms = "layout(packed) uniform MovitUniforms {\n" + frag_shader_uniforms + "};\n";
	}

	// Samplers must be outside the UBO block.
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		Effect *effect = node->effect;
		const string effect_id = phase->effect_ids[node];
		extract_uniform_declarations(effect->uniforms_sampler2d, "sampler2D", effect_id, /*in_ubo_block=*/false, &phase->uniforms_sampler2d, &frag_shader_uniforms);
	}

	frag_shader = frag_shader_header + frag_shader_uniforms + frag_shader;

	string vert_shader = read_version_dependent_file("vs", "vert");

	// If we're the last phase and need to flip the picture to compensate for
	// the origin, tell the vertex shader so.
	if (phase->output_node->outgoing_links.empty() && output_origin == OUTPUT_ORIGIN_TOP_LEFT) {
		const string needle = "#define FLIP_ORIGIN 0";
		size_t pos = vert_shader.find(needle);
		assert(pos != string::npos);

		vert_shader[pos + needle.size() - 1] = '1';
	}

	phase->glsl_program_num = resource_pool->compile_glsl_program(vert_shader, frag_shader, frag_shader_outputs);
	GLint position_attribute_index = glGetAttribLocation(phase->glsl_program_num, "position");
	GLint texcoord_attribute_index = glGetAttribLocation(phase->glsl_program_num, "texcoord");
	if (position_attribute_index != -1) {
		phase->attribute_indexes.insert(position_attribute_index);
	}
	if (texcoord_attribute_index != -1) {
		phase->attribute_indexes.insert(texcoord_attribute_index);
	}

	// Create an UBO for holding the uniforms. This UBO will be updated each frame.
	// TODO: Delete the block on destruction.
	phase->uniform_block_index = glGetUniformBlockIndex(phase->glsl_program_num, "MovitUniforms");
	if (phase->uniform_block_index != GL_INVALID_INDEX) {
		glGenBuffers(1, &phase->ubo);
		check_error();
		GLsizei block_size;
		glGetActiveUniformBlockiv(
			phase->glsl_program_num, phase->uniform_block_index,
			GL_UNIFORM_BLOCK_DATA_SIZE, &block_size);
		check_error();
		phase->ubo_data.resize(block_size);

		glBindBuffer(GL_UNIFORM_BUFFER, phase->ubo);
		check_error();
		glBufferData(GL_UNIFORM_BUFFER, block_size, NULL, GL_DYNAMIC_DRAW);
		check_error();

		// Associate the uniform block with binding point 0,
		// and attach the UBO to that binding point.
		glUniformBlockBinding(phase->glsl_program_num, phase->uniform_block_index, 0);
		check_error();
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, phase->ubo);
		check_error();
	} else {
		phase->ubo = GL_INVALID_INDEX;
	}

	// Collect the resulting location numbers for each uniform.
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_sampler2d);
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_bool);
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_int);
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_float);
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_vec2);
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_vec3);
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_vec4);
	collect_uniform_locations(phase->glsl_program_num, &phase->uniforms_mat3);

	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	check_error();
}

// Construct GLSL programs, starting at the given effect and following
// the chain from there. We end a program every time we come to an effect
// marked as "needs texture bounce", one that is used by multiple other
// effects, every time we need to bounce due to output size change
// (not all size changes require ending), and of course at the end.
//
// We follow a quite simple depth-first search from the output, although
// without recursing explicitly within each phase.
Phase *EffectChain::construct_phase(Node *output, map<Node *, Phase *> *completed_effects)
{
	if (completed_effects->count(output)) {
		return (*completed_effects)[output];
	}

	Phase *phase = new Phase;
	phase->output_node = output;

	// If the output effect has one-to-one sampling, we try to trace this
	// status down through the dependency chain. This is important in case
	// we hit an effect that changes output size (and not sets a virtual
	// output size); if we have one-to-one sampling, we don't have to break
	// the phase.
	output->one_to_one_sampling = output->effect->one_to_one_sampling();

	// Effects that we have yet to calculate, but that we know should
	// be in the current phase.
	stack<Node *> effects_todo_this_phase;
	effects_todo_this_phase.push(output);

	while (!effects_todo_this_phase.empty()) {
		Node *node = effects_todo_this_phase.top();
		effects_todo_this_phase.pop();

		if (node->effect->needs_mipmaps()) {
			node->needs_mipmaps = true;
		}

		// This should currently only happen for effects that are inputs
		// (either true inputs or phase outputs). We special-case inputs,
		// and then deduplicate phase outputs below.
		if (node->effect->num_inputs() == 0) {
			if (find(phase->effects.begin(), phase->effects.end(), node) != phase->effects.end()) {
				continue;
			}
		} else {
			assert(completed_effects->count(node) == 0);
		}

		phase->effects.push_back(node);

		// Find all the dependencies of this effect, and add them to the stack.
		vector<Node *> deps = node->incoming_links;
		assert(node->effect->num_inputs() == deps.size());
		for (unsigned i = 0; i < deps.size(); ++i) {
			bool start_new_phase = false;

			if (node->effect->needs_texture_bounce() &&
			    !deps[i]->effect->is_single_texture() &&
			    !deps[i]->effect->override_disable_bounce()) {
				start_new_phase = true;
			}

			// Propagate information about needing mipmaps down the chain,
			// breaking the phase if we notice an incompatibility.
			//
			// Note that we cannot do this propagation as a normal pass,
			// because it needs information about where the phases end
			// (we should not propagate the flag across phases).
			if (node->needs_mipmaps) {
				if (deps[i]->effect->num_inputs() == 0) {
					Input *input = static_cast<Input *>(deps[i]->effect);
					start_new_phase |= !input->can_supply_mipmaps();
				} else {
					deps[i]->needs_mipmaps = true;
				}
			}

			if (deps[i]->outgoing_links.size() > 1) {
				if (!deps[i]->effect->is_single_texture()) {
					// More than one effect uses this as the input,
					// and it is not a texture itself.
					// The easiest thing to do (and probably also the safest
					// performance-wise in most cases) is to bounce it to a texture
					// and then let the next passes read from that.
					start_new_phase = true;
				} else {
					assert(deps[i]->effect->num_inputs() == 0);

					// For textures, we try to be slightly more clever;
					// if none of our outputs need a bounce, we don't bounce
					// but instead simply use the effect many times.
					//
					// Strictly speaking, we could bounce it for some outputs
					// and use it directly for others, but the processing becomes
					// somewhat simpler if the effect is only used in one such way.
					for (unsigned j = 0; j < deps[i]->outgoing_links.size(); ++j) {
						Node *rdep = deps[i]->outgoing_links[j];
						start_new_phase |= rdep->effect->needs_texture_bounce();
					}
				}
			}

			if (deps[i]->effect->sets_virtual_output_size()) {
				assert(deps[i]->effect->changes_output_size());
				// If the next effect sets a virtual size to rely on OpenGL's
				// bilinear sampling, we'll really need to break the phase here.
				start_new_phase = true;
			} else if (deps[i]->effect->changes_output_size() && !node->one_to_one_sampling) {
				// If the next effect changes size and we don't have one-to-one sampling,
				// we also need to break here.
				start_new_phase = true;
			}

			if (start_new_phase) {
				phase->inputs.push_back(construct_phase(deps[i], completed_effects));
			} else {
				effects_todo_this_phase.push(deps[i]);

				// Propagate the one-to-one status down through the dependency.
				deps[i]->one_to_one_sampling = node->one_to_one_sampling &&
					deps[i]->effect->one_to_one_sampling();
			}
		}
	}

	// No more effects to do this phase. Take all the ones we have,
	// and create a GLSL program for it.
	assert(!phase->effects.empty());

	// Deduplicate the inputs, but don't change the ordering e.g. by sorting;
	// that would be nondeterministic and thus reduce cacheability.
	// TODO: Make this even more deterministic.
	vector<Phase *> dedup_inputs;
	set<Phase *> seen_inputs;
	for (size_t i = 0; i < phase->inputs.size(); ++i) {
		if (seen_inputs.insert(phase->inputs[i]).second) {
			dedup_inputs.push_back(phase->inputs[i]);
		}
	}
	swap(phase->inputs, dedup_inputs);

	// Allocate samplers for each input.
	phase->input_samplers.resize(phase->inputs.size());

	// We added the effects from the output and back, but we need to output
	// them in topological sort order in the shader.
	phase->effects = topological_sort(phase->effects);

	// Figure out if we need mipmaps or not, and if so, tell the inputs that.
	phase->input_needs_mipmaps = false;
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		phase->input_needs_mipmaps |= node->effect->needs_mipmaps();
	}
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		if (node->effect->num_inputs() == 0) {
			Input *input = static_cast<Input *>(node->effect);
			assert(!phase->input_needs_mipmaps || input->can_supply_mipmaps());
			CHECK(input->set_int("needs_mipmaps", phase->input_needs_mipmaps));
		}
	}

	// Tell each node which phase it ended up in, so that the unit test
	// can check that the phases were split in the right place.
	// Note that this ignores that effects may be part of multiple phases;
	// if the unit tests need to test such cases, we'll reconsider.
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		phase->effects[i]->containing_phase = phase;
	}

	// Actually make the shader for this phase.
	compile_glsl_program(phase);

	// Initialize timers.
	if (movit_timer_queries_supported) {
		phase->time_elapsed_ns = 0;
		phase->num_measured_iterations = 0;
	}

	assert(completed_effects->count(output) == 0);
	completed_effects->insert(make_pair(output, phase));
	phases.push_back(phase);
	return phase;
}

void EffectChain::output_dot(const char *filename)
{
	if (movit_debug_level != MOVIT_DEBUG_ON) {
		return;
	}

	FILE *fp = fopen(filename, "w");
	if (fp == NULL) {
		perror(filename);
		exit(1);
	}

	fprintf(fp, "digraph G {\n");
	fprintf(fp, "  output [shape=box label=\"(output)\"];\n");
	for (unsigned i = 0; i < nodes.size(); ++i) {
		// Find out which phase this event belongs to.
		vector<int> in_phases;
		for (unsigned j = 0; j < phases.size(); ++j) {
			const Phase* p = phases[j];
			if (find(p->effects.begin(), p->effects.end(), nodes[i]) != p->effects.end()) {
				in_phases.push_back(j);
			}
		}

		if (in_phases.empty()) {
			fprintf(fp, "  n%ld [label=\"%s\"];\n", (long)nodes[i], nodes[i]->effect->effect_type_id().c_str());
		} else if (in_phases.size() == 1) {
			fprintf(fp, "  n%ld [label=\"%s\" style=\"filled\" fillcolor=\"/accent8/%d\"];\n",
				(long)nodes[i], nodes[i]->effect->effect_type_id().c_str(),
				(in_phases[0] % 8) + 1);
		} else {
			// If we had new enough Graphviz, style="wedged" would probably be ideal here.
			// But alas.
			fprintf(fp, "  n%ld [label=\"%s [in multiple phases]\" style=\"filled\" fillcolor=\"/accent8/%d\"];\n",
				(long)nodes[i], nodes[i]->effect->effect_type_id().c_str(),
				(in_phases[0] % 8) + 1);
		}

		char from_node_id[256];
		snprintf(from_node_id, 256, "n%ld", (long)nodes[i]);

		for (unsigned j = 0; j < nodes[i]->outgoing_links.size(); ++j) {
			char to_node_id[256];
			snprintf(to_node_id, 256, "n%ld", (long)nodes[i]->outgoing_links[j]);

			vector<string> labels = get_labels_for_edge(nodes[i], nodes[i]->outgoing_links[j]);
			output_dot_edge(fp, from_node_id, to_node_id, labels);
		}

		if (nodes[i]->outgoing_links.empty() && !nodes[i]->disabled) {
			// Output node.
			vector<string> labels = get_labels_for_edge(nodes[i], NULL);
			output_dot_edge(fp, from_node_id, "output", labels);
		}
	}
	fprintf(fp, "}\n");

	fclose(fp);
}

vector<string> EffectChain::get_labels_for_edge(const Node *from, const Node *to)
{
	vector<string> labels;

	if (to != NULL && to->effect->needs_texture_bounce()) {
		labels.push_back("needs_bounce");
	}
	if (from->effect->changes_output_size()) {
		labels.push_back("resize");
	}

	switch (from->output_color_space) {
	case COLORSPACE_INVALID:
		labels.push_back("spc[invalid]");
		break;
	case COLORSPACE_REC_601_525:
		labels.push_back("spc[rec601-525]");
		break;
	case COLORSPACE_REC_601_625:
		labels.push_back("spc[rec601-625]");
		break;
	default:
		break;
	}

	switch (from->output_gamma_curve) {
	case GAMMA_INVALID:
		labels.push_back("gamma[invalid]");
		break;
	case GAMMA_sRGB:
		labels.push_back("gamma[sRGB]");
		break;
	case GAMMA_REC_601:  // and GAMMA_REC_709
		labels.push_back("gamma[rec601/709]");
		break;
	default:
		break;
	}

	switch (from->output_alpha_type) {
	case ALPHA_INVALID:
		labels.push_back("alpha[invalid]");
		break;
	case ALPHA_BLANK:
		labels.push_back("alpha[blank]");
		break;
	case ALPHA_POSTMULTIPLIED:
		labels.push_back("alpha[postmult]");
		break;
	default:
		break;
	}

	return labels;
}

void EffectChain::output_dot_edge(FILE *fp,
                                  const string &from_node_id,
                                  const string &to_node_id,
                                  const vector<string> &labels)
{
	if (labels.empty()) {
		fprintf(fp, "  %s -> %s;\n", from_node_id.c_str(), to_node_id.c_str());
	} else {
		string label = labels[0];
		for (unsigned k = 1; k < labels.size(); ++k) {
			label += ", " + labels[k];
		}
		fprintf(fp, "  %s -> %s [label=\"%s\"];\n", from_node_id.c_str(), to_node_id.c_str(), label.c_str());
	}
}

void EffectChain::size_rectangle_to_fit(unsigned width, unsigned height, unsigned *output_width, unsigned *output_height)
{
	unsigned scaled_width, scaled_height;

	if (float(width) * aspect_denom >= float(height) * aspect_nom) {
		// Same aspect, or W/H > aspect (image is wider than the frame).
		// In either case, keep width, and adjust height.
		scaled_width = width;
		scaled_height = lrintf(width * aspect_denom / aspect_nom);
	} else {
		// W/H < aspect (image is taller than the frame), so keep height,
		// and adjust width.
		scaled_width = lrintf(height * aspect_nom / aspect_denom);
		scaled_height = height;
	}

	// We should be consistently larger or smaller then the existing choice,
	// since we have the same aspect.
	assert(!(scaled_width < *output_width && scaled_height > *output_height));
	assert(!(scaled_height < *output_height && scaled_width > *output_width));

	if (scaled_width >= *output_width && scaled_height >= *output_height) {
		*output_width = scaled_width;
		*output_height = scaled_height;
	}
}

// Propagate input texture sizes throughout, and inform effects downstream.
// (Like a lot of other code, we depend on effects being in topological order.)
void EffectChain::inform_input_sizes(Phase *phase)
{
	// All effects that have a defined size (inputs and RTT inputs)
	// get that. Reset all others.
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		if (node->effect->num_inputs() == 0) {
			Input *input = static_cast<Input *>(node->effect);
			node->output_width = input->get_width();
			node->output_height = input->get_height();
			assert(node->output_width != 0);
			assert(node->output_height != 0);
		} else {
			node->output_width = node->output_height = 0;
		}
	}
	for (unsigned i = 0; i < phase->inputs.size(); ++i) {
		Phase *input = phase->inputs[i];
		input->output_node->output_width = input->virtual_output_width;
		input->output_node->output_height = input->virtual_output_height;
		assert(input->output_node->output_width != 0);
		assert(input->output_node->output_height != 0);
	}

	// Now propagate from the inputs towards the end, and inform as we go.
	// The rules are simple:
	//
	//   1. Don't touch effects that already have given sizes (ie., inputs
	//      or effects that change the output size).
	//   2. If all of your inputs have the same size, that will be your output size.
	//   3. Otherwise, your output size is 0x0.
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		if (node->effect->num_inputs() == 0) {
			continue;
		}
		unsigned this_output_width = 0;
		unsigned this_output_height = 0;
		for (unsigned j = 0; j < node->incoming_links.size(); ++j) {
			Node *input = node->incoming_links[j];
			node->effect->inform_input_size(j, input->output_width, input->output_height);
			if (j == 0) {
				this_output_width = input->output_width;
				this_output_height = input->output_height;
			} else if (input->output_width != this_output_width || input->output_height != this_output_height) {
				// Inputs disagree.
				this_output_width = 0;
				this_output_height = 0;
			}
		}
		if (node->effect->changes_output_size()) {
			// We cannot call get_output_size() before we've done inform_input_size()
			// on all inputs.
			unsigned real_width, real_height;
			node->effect->get_output_size(&real_width, &real_height,
			                              &node->output_width, &node->output_height);
			assert(node->effect->sets_virtual_output_size() ||
			       (real_width == node->output_width &&
			        real_height == node->output_height));
		} else {
			node->output_width = this_output_width;
			node->output_height = this_output_height;
		}
	}
}

// Note: You should call inform_input_sizes() before this, as the last effect's
// desired output size might change based on the inputs.
void EffectChain::find_output_size(Phase *phase)
{
	Node *output_node = phase->effects.back();

	// If the last effect explicitly sets an output size, use that.
	if (output_node->effect->changes_output_size()) {
		output_node->effect->get_output_size(&phase->output_width, &phase->output_height,
		                                     &phase->virtual_output_width, &phase->virtual_output_height);
		assert(output_node->effect->sets_virtual_output_size() ||
		       (phase->output_width == phase->virtual_output_width &&
			phase->output_height == phase->virtual_output_height));
		return;
	}

	// If all effects have the same size, use that.
	unsigned output_width = 0, output_height = 0;
	bool all_inputs_same_size = true;

	for (unsigned i = 0; i < phase->inputs.size(); ++i) {
		Phase *input = phase->inputs[i];
		assert(input->output_width != 0);
		assert(input->output_height != 0);
		if (output_width == 0 && output_height == 0) {
			output_width = input->virtual_output_width;
			output_height = input->virtual_output_height;
		} else if (output_width != input->virtual_output_width ||
		           output_height != input->virtual_output_height) {
			all_inputs_same_size = false;
		}
	}
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Effect *effect = phase->effects[i]->effect;
		if (effect->num_inputs() != 0) {
			continue;
		}

		Input *input = static_cast<Input *>(effect);
		if (output_width == 0 && output_height == 0) {
			output_width = input->get_width();
			output_height = input->get_height();
		} else if (output_width != input->get_width() ||
		           output_height != input->get_height()) {
			all_inputs_same_size = false;
		}
	}

	if (all_inputs_same_size) {
		assert(output_width != 0);
		assert(output_height != 0);
		phase->virtual_output_width = phase->output_width = output_width;
		phase->virtual_output_height = phase->output_height = output_height;
		return;
	}

	// If not, fit all the inputs into the current aspect, and select the largest one. 
	output_width = 0;
	output_height = 0;
	for (unsigned i = 0; i < phase->inputs.size(); ++i) {
		Phase *input = phase->inputs[i];
		assert(input->output_width != 0);
		assert(input->output_height != 0);
		size_rectangle_to_fit(input->output_width, input->output_height, &output_width, &output_height);
	}
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Effect *effect = phase->effects[i]->effect;
		if (effect->num_inputs() != 0) {
			continue;
		}

		Input *input = static_cast<Input *>(effect);
		size_rectangle_to_fit(input->get_width(), input->get_height(), &output_width, &output_height);
	}
	assert(output_width != 0);
	assert(output_height != 0);
	phase->virtual_output_width = phase->output_width = output_width;
	phase->virtual_output_height = phase->output_height = output_height;
}

void EffectChain::sort_all_nodes_topologically()
{
	nodes = topological_sort(nodes);
}

vector<Node *> EffectChain::topological_sort(const vector<Node *> &nodes)
{
	set<Node *> nodes_left_to_visit(nodes.begin(), nodes.end());
	vector<Node *> sorted_list;
	for (unsigned i = 0; i < nodes.size(); ++i) {
		topological_sort_visit_node(nodes[i], &nodes_left_to_visit, &sorted_list);
	}
	reverse(sorted_list.begin(), sorted_list.end());
	return sorted_list;
}

void EffectChain::topological_sort_visit_node(Node *node, set<Node *> *nodes_left_to_visit, vector<Node *> *sorted_list)
{
	if (nodes_left_to_visit->count(node) == 0) {
		return;
	}
	nodes_left_to_visit->erase(node);
	for (unsigned i = 0; i < node->outgoing_links.size(); ++i) {
		topological_sort_visit_node(node->outgoing_links[i], nodes_left_to_visit, sorted_list);
	}
	sorted_list->push_back(node);
}

void EffectChain::find_color_spaces_for_inputs()
{
	for (unsigned i = 0; i < nodes.size(); ++i) {
		Node *node = nodes[i];
		if (node->disabled) {
			continue;
		}
		if (node->incoming_links.size() == 0) {
			Input *input = static_cast<Input *>(node->effect);
			node->output_color_space = input->get_color_space();
			node->output_gamma_curve = input->get_gamma_curve();

			Effect::AlphaHandling alpha_handling = input->alpha_handling();
			switch (alpha_handling) {
			case Effect::OUTPUT_BLANK_ALPHA:
				node->output_alpha_type = ALPHA_BLANK;
				break;
			case Effect::INPUT_AND_OUTPUT_PREMULTIPLIED_ALPHA:
				node->output_alpha_type = ALPHA_PREMULTIPLIED;
				break;
			case Effect::OUTPUT_POSTMULTIPLIED_ALPHA:
				node->output_alpha_type = ALPHA_POSTMULTIPLIED;
				break;
			case Effect::INPUT_PREMULTIPLIED_ALPHA_KEEP_BLANK:
			case Effect::DONT_CARE_ALPHA_TYPE:
			default:
				assert(false);
			}

			if (node->output_alpha_type == ALPHA_PREMULTIPLIED) {
				assert(node->output_gamma_curve == GAMMA_LINEAR);
			}
		}
	}
}

// Propagate gamma and color space information as far as we can in the graph.
// The rules are simple: Anything where all the inputs agree, get that as
// output as well. Anything else keeps having *_INVALID.
void EffectChain::propagate_gamma_and_color_space()
{
	// We depend on going through the nodes in order.
	sort_all_nodes_topologically();

	for (unsigned i = 0; i < nodes.size(); ++i) {
		Node *node = nodes[i];
		if (node->disabled) {
			continue;
		}
		assert(node->incoming_links.size() == node->effect->num_inputs());
		if (node->incoming_links.size() == 0) {
			assert(node->output_color_space != COLORSPACE_INVALID);
			assert(node->output_gamma_curve != GAMMA_INVALID);
			continue;
		}

		Colorspace color_space = node->incoming_links[0]->output_color_space;
		GammaCurve gamma_curve = node->incoming_links[0]->output_gamma_curve;
		for (unsigned j = 1; j < node->incoming_links.size(); ++j) {
			if (node->incoming_links[j]->output_color_space != color_space) {
				color_space = COLORSPACE_INVALID;
			}
			if (node->incoming_links[j]->output_gamma_curve != gamma_curve) {
				gamma_curve = GAMMA_INVALID;
			}
		}

		// The conversion effects already have their outputs set correctly,
		// so leave them alone.
		if (node->effect->effect_type_id() != "ColorspaceConversionEffect") {
			node->output_color_space = color_space;
		}		
		if (node->effect->effect_type_id() != "GammaCompressionEffect" &&
		    node->effect->effect_type_id() != "GammaExpansionEffect") {
			node->output_gamma_curve = gamma_curve;
		}		
	}
}

// Propagate alpha information as far as we can in the graph.
// Similar to propagate_gamma_and_color_space().
void EffectChain::propagate_alpha()
{
	// We depend on going through the nodes in order.
	sort_all_nodes_topologically();

	for (unsigned i = 0; i < nodes.size(); ++i) {
		Node *node = nodes[i];
		if (node->disabled) {
			continue;
		}
		assert(node->incoming_links.size() == node->effect->num_inputs());
		if (node->incoming_links.size() == 0) {
			assert(node->output_alpha_type != ALPHA_INVALID);
			continue;
		}

		// The alpha multiplication/division effects are special cases.
		if (node->effect->effect_type_id() == "AlphaMultiplicationEffect") {
			assert(node->incoming_links.size() == 1);
			assert(node->incoming_links[0]->output_alpha_type == ALPHA_POSTMULTIPLIED);
			node->output_alpha_type = ALPHA_PREMULTIPLIED;
			continue;
		}
		if (node->effect->effect_type_id() == "AlphaDivisionEffect") {
			assert(node->incoming_links.size() == 1);
			assert(node->incoming_links[0]->output_alpha_type == ALPHA_PREMULTIPLIED);
			node->output_alpha_type = ALPHA_POSTMULTIPLIED;
			continue;
		}

		// GammaCompressionEffect and GammaExpansionEffect are also a special case,
		// because they are the only one that _need_ postmultiplied alpha.
		if (node->effect->effect_type_id() == "GammaCompressionEffect" ||
		    node->effect->effect_type_id() == "GammaExpansionEffect") {
			assert(node->incoming_links.size() == 1);
			if (node->incoming_links[0]->output_alpha_type == ALPHA_BLANK) {
				node->output_alpha_type = ALPHA_BLANK;
			} else if (node->incoming_links[0]->output_alpha_type == ALPHA_POSTMULTIPLIED) {
				node->output_alpha_type = ALPHA_POSTMULTIPLIED;
			} else {
				node->output_alpha_type = ALPHA_INVALID;
			}
			continue;
		}

		// Only inputs can have unconditional alpha output (OUTPUT_BLANK_ALPHA
		// or OUTPUT_POSTMULTIPLIED_ALPHA), and they have already been
		// taken care of above. Rationale: Even if you could imagine
		// e.g. an effect that took in an image and set alpha=1.0
		// unconditionally, it wouldn't make any sense to have it as
		// e.g. OUTPUT_BLANK_ALPHA, since it wouldn't know whether it
		// got its input pre- or postmultiplied, so it wouldn't know
		// whether to divide away the old alpha or not.
		Effect::AlphaHandling alpha_handling = node->effect->alpha_handling();
		assert(alpha_handling == Effect::INPUT_AND_OUTPUT_PREMULTIPLIED_ALPHA ||
		       alpha_handling == Effect::INPUT_PREMULTIPLIED_ALPHA_KEEP_BLANK ||
		       alpha_handling == Effect::DONT_CARE_ALPHA_TYPE);

		// If the node has multiple inputs, check that they are all valid and
		// the same.
		bool any_invalid = false;
		bool any_premultiplied = false;
		bool any_postmultiplied = false;

		for (unsigned j = 0; j < node->incoming_links.size(); ++j) {
			switch (node->incoming_links[j]->output_alpha_type) {
			case ALPHA_INVALID:
				any_invalid = true;
				break;
			case ALPHA_BLANK:
				// Blank is good as both pre- and postmultiplied alpha,
				// so just ignore it.
				break;
			case ALPHA_PREMULTIPLIED:
				any_premultiplied = true;
				break;
			case ALPHA_POSTMULTIPLIED:
				any_postmultiplied = true;
				break;
			default:
				assert(false);
			}
		}

		if (any_invalid) {
			node->output_alpha_type = ALPHA_INVALID;
			continue;
		}

		// Inputs must be of the same type.
		if (any_premultiplied && any_postmultiplied) {
			node->output_alpha_type = ALPHA_INVALID;
			continue;
		}

		if (alpha_handling == Effect::INPUT_AND_OUTPUT_PREMULTIPLIED_ALPHA ||
		    alpha_handling == Effect::INPUT_PREMULTIPLIED_ALPHA_KEEP_BLANK) {
			// If the effect has asked for premultiplied alpha, check that it has got it.
			if (any_postmultiplied) {
				node->output_alpha_type = ALPHA_INVALID;
			} else if (!any_premultiplied &&
			           alpha_handling == Effect::INPUT_PREMULTIPLIED_ALPHA_KEEP_BLANK) {
				// Blank input alpha, and the effect preserves blank alpha.
				node->output_alpha_type = ALPHA_BLANK;
			} else {
				node->output_alpha_type = ALPHA_PREMULTIPLIED;
			}
		} else {
			// OK, all inputs are the same, and this effect is not going
			// to change it.
			assert(alpha_handling == Effect::DONT_CARE_ALPHA_TYPE);
			if (any_premultiplied) {
				node->output_alpha_type = ALPHA_PREMULTIPLIED;
			} else if (any_postmultiplied) {
				node->output_alpha_type = ALPHA_POSTMULTIPLIED;
			} else {
				node->output_alpha_type = ALPHA_BLANK;
			}
		}
	}
}

bool EffectChain::node_needs_colorspace_fix(Node *node)
{
	if (node->disabled) {
		return false;
	}
	if (node->effect->num_inputs() == 0) {
		return false;
	}

	// propagate_gamma_and_color_space() has already set our output
	// to COLORSPACE_INVALID if the inputs differ, so we can rely on that.
	if (node->output_color_space == COLORSPACE_INVALID) {
		return true;
	}
	return (node->effect->needs_srgb_primaries() && node->output_color_space != COLORSPACE_sRGB);
}

// Fix up color spaces so that there are no COLORSPACE_INVALID nodes left in
// the graph. Our strategy is not always optimal, but quite simple:
// Find an effect that's as early as possible where the inputs are of
// unacceptable colorspaces (that is, either different, or, if the effect only
// wants sRGB, not sRGB.) Add appropriate conversions on all its inputs,
// propagate the information anew, and repeat until there are no more such
// effects.
void EffectChain::fix_internal_color_spaces()
{
	unsigned colorspace_propagation_pass = 0;
	bool found_any;
	do {
		found_any = false;
		for (unsigned i = 0; i < nodes.size(); ++i) {
			Node *node = nodes[i];
			if (!node_needs_colorspace_fix(node)) {
				continue;
			}

			// Go through each input that is not sRGB, and insert
			// a colorspace conversion after it.
			for (unsigned j = 0; j < node->incoming_links.size(); ++j) {
				Node *input = node->incoming_links[j];
				assert(input->output_color_space != COLORSPACE_INVALID);
				if (input->output_color_space == COLORSPACE_sRGB) {
					continue;
				}
				Node *conversion = add_node(new ColorspaceConversionEffect());
				CHECK(conversion->effect->set_int("source_space", input->output_color_space));
				CHECK(conversion->effect->set_int("destination_space", COLORSPACE_sRGB));
				conversion->output_color_space = COLORSPACE_sRGB;
				replace_sender(input, conversion);
				connect_nodes(input, conversion);
			}

			// Re-sort topologically, and propagate the new information.
			propagate_gamma_and_color_space();
			
			found_any = true;
			break;
		}
	
		char filename[256];
		sprintf(filename, "step5-colorspacefix-iter%u.dot", ++colorspace_propagation_pass);
		output_dot(filename);
		assert(colorspace_propagation_pass < 100);
	} while (found_any);

	for (unsigned i = 0; i < nodes.size(); ++i) {
		Node *node = nodes[i];
		if (node->disabled) {
			continue;
		}
		assert(node->output_color_space != COLORSPACE_INVALID);
	}
}

bool EffectChain::node_needs_alpha_fix(Node *node)
{
	if (node->disabled) {
		return false;
	}

	// propagate_alpha() has already set our output to ALPHA_INVALID if the
	// inputs differ or we are otherwise in mismatch, so we can rely on that.
	return (node->output_alpha_type == ALPHA_INVALID);
}

// Fix up alpha so that there are no ALPHA_INVALID nodes left in
// the graph. Similar to fix_internal_color_spaces().
void EffectChain::fix_internal_alpha(unsigned step)
{
	unsigned alpha_propagation_pass = 0;
	bool found_any;
	do {
		found_any = false;
		for (unsigned i = 0; i < nodes.size(); ++i) {
			Node *node = nodes[i];
			if (!node_needs_alpha_fix(node)) {
				continue;
			}

			// If we need to fix up GammaExpansionEffect, then clearly something
			// is wrong, since the combination of premultiplied alpha and nonlinear inputs
			// is meaningless.
			assert(node->effect->effect_type_id() != "GammaExpansionEffect");

			AlphaType desired_type = ALPHA_PREMULTIPLIED;

			// GammaCompressionEffect is special; it needs postmultiplied alpha.
			if (node->effect->effect_type_id() == "GammaCompressionEffect") {
				assert(node->incoming_links.size() == 1);
				assert(node->incoming_links[0]->output_alpha_type == ALPHA_PREMULTIPLIED);
				desired_type = ALPHA_POSTMULTIPLIED;
			}

			// Go through each input that is not premultiplied alpha, and insert
			// a conversion before it.
			for (unsigned j = 0; j < node->incoming_links.size(); ++j) {
				Node *input = node->incoming_links[j];
				assert(input->output_alpha_type != ALPHA_INVALID);
				if (input->output_alpha_type == desired_type ||
				    input->output_alpha_type == ALPHA_BLANK) {
					continue;
				}
				Node *conversion;
				if (desired_type == ALPHA_PREMULTIPLIED) {
					conversion = add_node(new AlphaMultiplicationEffect());
				} else {
					conversion = add_node(new AlphaDivisionEffect());
				}
				conversion->output_alpha_type = desired_type;
				replace_sender(input, conversion);
				connect_nodes(input, conversion);
			}

			// Re-sort topologically, and propagate the new information.
			propagate_gamma_and_color_space();
			propagate_alpha();
			
			found_any = true;
			break;
		}
	
		char filename[256];
		sprintf(filename, "step%u-alphafix-iter%u.dot", step, ++alpha_propagation_pass);
		output_dot(filename);
		assert(alpha_propagation_pass < 100);
	} while (found_any);

	for (unsigned i = 0; i < nodes.size(); ++i) {
		Node *node = nodes[i];
		if (node->disabled) {
			continue;
		}
		assert(node->output_alpha_type != ALPHA_INVALID);
	}
}

// Make so that the output is in the desired color space.
void EffectChain::fix_output_color_space()
{
	Node *output = find_output_node();
	if (output->output_color_space != output_format.color_space) {
		Node *conversion = add_node(new ColorspaceConversionEffect());
		CHECK(conversion->effect->set_int("source_space", output->output_color_space));
		CHECK(conversion->effect->set_int("destination_space", output_format.color_space));
		conversion->output_color_space = output_format.color_space;
		connect_nodes(output, conversion);
		propagate_alpha();
		propagate_gamma_and_color_space();
	}
}

// Make so that the output is in the desired pre-/postmultiplication alpha state.
void EffectChain::fix_output_alpha()
{
	Node *output = find_output_node();
	assert(output->output_alpha_type != ALPHA_INVALID);
	if (output->output_alpha_type == ALPHA_BLANK) {
		// No alpha output, so we don't care.
		return;
	}
	if (output->output_alpha_type == ALPHA_PREMULTIPLIED &&
	    output_alpha_format == OUTPUT_ALPHA_FORMAT_POSTMULTIPLIED) {
		Node *conversion = add_node(new AlphaDivisionEffect());
		connect_nodes(output, conversion);
		propagate_alpha();
		propagate_gamma_and_color_space();
	}
	if (output->output_alpha_type == ALPHA_POSTMULTIPLIED &&
	    output_alpha_format == OUTPUT_ALPHA_FORMAT_PREMULTIPLIED) {
		Node *conversion = add_node(new AlphaMultiplicationEffect());
		connect_nodes(output, conversion);
		propagate_alpha();
		propagate_gamma_and_color_space();
	}
}

bool EffectChain::node_needs_gamma_fix(Node *node)
{
	if (node->disabled) {
		return false;
	}

	// Small hack since the output is not an explicit node:
	// If we are the last node and our output is in the wrong
	// space compared to EffectChain's output, we need to fix it.
	// This will only take us to linear, but fix_output_gamma()
	// will come and take us to the desired output gamma
	// if it is needed.
	//
	// This needs to be before everything else, since it could
	// even apply to inputs (if they are the only effect).
	if (node->outgoing_links.empty() &&
	    node->output_gamma_curve != output_format.gamma_curve &&
	    node->output_gamma_curve != GAMMA_LINEAR) {
		return true;
	}

	if (node->effect->num_inputs() == 0) {
		return false;
	}

	// propagate_gamma_and_color_space() has already set our output
	// to GAMMA_INVALID if the inputs differ, so we can rely on that,
	// except for GammaCompressionEffect.
	if (node->output_gamma_curve == GAMMA_INVALID) {
		return true;
	}
	if (node->effect->effect_type_id() == "GammaCompressionEffect") {
		assert(node->incoming_links.size() == 1);
		return node->incoming_links[0]->output_gamma_curve != GAMMA_LINEAR;
	}

	return (node->effect->needs_linear_light() && node->output_gamma_curve != GAMMA_LINEAR);
}

// Very similar to fix_internal_color_spaces(), but for gamma.
// There is one difference, though; before we start adding conversion nodes,
// we see if we can get anything out of asking the sources to deliver
// linear gamma directly. fix_internal_gamma_by_asking_inputs()
// does that part, while fix_internal_gamma_by_inserting_nodes()
// inserts nodes as needed afterwards.
void EffectChain::fix_internal_gamma_by_asking_inputs(unsigned step)
{
	unsigned gamma_propagation_pass = 0;
	bool found_any;
	do {
		found_any = false;
		for (unsigned i = 0; i < nodes.size(); ++i) {
			Node *node = nodes[i];
			if (!node_needs_gamma_fix(node)) {
				continue;
			}

			// See if all inputs can give us linear gamma. If not, leave it.
			vector<Node *> nonlinear_inputs;
			find_all_nonlinear_inputs(node, &nonlinear_inputs);
			assert(!nonlinear_inputs.empty());

			bool all_ok = true;
			for (unsigned i = 0; i < nonlinear_inputs.size(); ++i) {
				Input *input = static_cast<Input *>(nonlinear_inputs[i]->effect);
				all_ok &= input->can_output_linear_gamma();
			}

			if (!all_ok) {
				continue;
			}

			for (unsigned i = 0; i < nonlinear_inputs.size(); ++i) {
				CHECK(nonlinear_inputs[i]->effect->set_int("output_linear_gamma", 1));
				nonlinear_inputs[i]->output_gamma_curve = GAMMA_LINEAR;
			}

			// Re-sort topologically, and propagate the new information.
			propagate_gamma_and_color_space();
			
			found_any = true;
			break;
		}
	
		char filename[256];
		sprintf(filename, "step%u-gammafix-iter%u.dot", step, ++gamma_propagation_pass);
		output_dot(filename);
		assert(gamma_propagation_pass < 100);
	} while (found_any);
}

void EffectChain::fix_internal_gamma_by_inserting_nodes(unsigned step)
{
	unsigned gamma_propagation_pass = 0;
	bool found_any;
	do {
		found_any = false;
		for (unsigned i = 0; i < nodes.size(); ++i) {
			Node *node = nodes[i];
			if (!node_needs_gamma_fix(node)) {
				continue;
			}

			// Special case: We could be an input and still be asked to
			// fix our gamma; if so, we should be the only node
			// (as node_needs_gamma_fix() would only return true in
			// for an input in that case). That means we should insert
			// a conversion node _after_ ourselves.
			if (node->incoming_links.empty()) {
				assert(node->outgoing_links.empty());
				Node *conversion = add_node(new GammaExpansionEffect());
				CHECK(conversion->effect->set_int("source_curve", node->output_gamma_curve));
				conversion->output_gamma_curve = GAMMA_LINEAR;
				connect_nodes(node, conversion);
			}

			// If not, go through each input that is not linear gamma,
			// and insert a gamma conversion after it.
			for (unsigned j = 0; j < node->incoming_links.size(); ++j) {
				Node *input = node->incoming_links[j];
				assert(input->output_gamma_curve != GAMMA_INVALID);
				if (input->output_gamma_curve == GAMMA_LINEAR) {
					continue;
				}
				Node *conversion = add_node(new GammaExpansionEffect());
				CHECK(conversion->effect->set_int("source_curve", input->output_gamma_curve));
				conversion->output_gamma_curve = GAMMA_LINEAR;
				replace_sender(input, conversion);
				connect_nodes(input, conversion);
			}

			// Re-sort topologically, and propagate the new information.
			propagate_alpha();
			propagate_gamma_and_color_space();
			
			found_any = true;
			break;
		}
	
		char filename[256];
		sprintf(filename, "step%u-gammafix-iter%u.dot", step, ++gamma_propagation_pass);
		output_dot(filename);
		assert(gamma_propagation_pass < 100);
	} while (found_any);

	for (unsigned i = 0; i < nodes.size(); ++i) {
		Node *node = nodes[i];
		if (node->disabled) {
			continue;
		}
		assert(node->output_gamma_curve != GAMMA_INVALID);
	}
}

// Make so that the output is in the desired gamma.
// Note that this assumes linear input gamma, so it might create the need
// for another pass of fix_internal_gamma().
void EffectChain::fix_output_gamma()
{
	Node *output = find_output_node();
	if (output->output_gamma_curve != output_format.gamma_curve) {
		Node *conversion = add_node(new GammaCompressionEffect());
		CHECK(conversion->effect->set_int("destination_curve", output_format.gamma_curve));
		conversion->output_gamma_curve = output_format.gamma_curve;
		connect_nodes(output, conversion);
	}
}

// If the user has requested Y'CbCr output, we need to do this conversion
// _after_ GammaCompressionEffect etc., but before dither (see below).
// This is because Y'CbCr, with the exception of a special optional mode
// in Rec. 2020 (which we currently don't support), is defined to work on
// gamma-encoded data.
void EffectChain::add_ycbcr_conversion_if_needed()
{
	assert(output_color_rgba || output_color_ycbcr);
	if (!output_color_ycbcr) {
		return;
	}
	Node *output = find_output_node();
	Node *ycbcr = add_node(new YCbCrConversionEffect(output_ycbcr_format));
	connect_nodes(output, ycbcr);
}
	
// If the user has requested dither, add a DitherEffect right at the end
// (after GammaCompressionEffect etc.). This needs to be done after everything else,
// since dither is about the only effect that can _not_ be done in linear space.
void EffectChain::add_dither_if_needed()
{
	if (num_dither_bits == 0) {
		return;
	}
	Node *output = find_output_node();
	Node *dither = add_node(new DitherEffect());
	CHECK(dither->effect->set_int("num_bits", num_dither_bits));
	connect_nodes(output, dither);

	dither_effect = dither->effect;
}

// Find the output node. This is, simply, one that has no outgoing links.
// If there are multiple ones, the graph is malformed (we do not support
// multiple outputs right now).
Node *EffectChain::find_output_node()
{
	vector<Node *> output_nodes;
	for (unsigned i = 0; i < nodes.size(); ++i) {
		Node *node = nodes[i];
		if (node->disabled) {
			continue;
		}
		if (node->outgoing_links.empty()) {
			output_nodes.push_back(node);
		}
	}
	assert(output_nodes.size() == 1);
	return output_nodes[0];
}

void EffectChain::finalize()
{
	// Output the graph as it is before we do any conversions on it.
	output_dot("step0-start.dot");

	// Give each effect in turn a chance to rewrite its own part of the graph.
	// Note that if more effects are added as part of this, they will be
	// picked up as part of the same for loop, since they are added at the end.
	for (unsigned i = 0; i < nodes.size(); ++i) {
		nodes[i]->effect->rewrite_graph(this, nodes[i]);
	}
	output_dot("step1-rewritten.dot");

	find_color_spaces_for_inputs();
	output_dot("step2-input-colorspace.dot");

	propagate_alpha();
	output_dot("step3-propagated-alpha.dot");

	propagate_gamma_and_color_space();
	output_dot("step4-propagated-all.dot");

	fix_internal_color_spaces();
	fix_internal_alpha(6);
	fix_output_color_space();
	output_dot("step7-output-colorspacefix.dot");
	fix_output_alpha();
	output_dot("step8-output-alphafix.dot");

	// Note that we need to fix gamma after colorspace conversion,
	// because colorspace conversions might create needs for gamma conversions.
	// Also, we need to run an extra pass of fix_internal_gamma() after 
	// fixing the output gamma, as we only have conversions to/from linear,
	// and fix_internal_alpha() since GammaCompressionEffect needs
	// postmultiplied input.
	fix_internal_gamma_by_asking_inputs(9);
	fix_internal_gamma_by_inserting_nodes(10);
	fix_output_gamma();
	output_dot("step11-output-gammafix.dot");
	propagate_alpha();
	output_dot("step12-output-alpha-propagated.dot");
	fix_internal_alpha(13);
	output_dot("step14-output-alpha-fixed.dot");
	fix_internal_gamma_by_asking_inputs(15);
	fix_internal_gamma_by_inserting_nodes(16);

	output_dot("step17-before-ycbcr.dot");
	add_ycbcr_conversion_if_needed();

	output_dot("step18-before-dither.dot");
	add_dither_if_needed();

	output_dot("step19-final.dot");
	
	// Construct all needed GLSL programs, starting at the output.
	// We need to keep track of which effects have already been computed,
	// as an effect with multiple users could otherwise be calculated
	// multiple times.
	map<Node *, Phase *> completed_effects;
	construct_phase(find_output_node(), &completed_effects);

	output_dot("step20-split-to-phases.dot");

	assert(phases[0]->inputs.empty());
	
	finalized = true;
}

void EffectChain::render_to_fbo(GLuint dest_fbo, unsigned width, unsigned height)
{
	assert(finalized);

	// This needs to be set anew, in case we are coming from a different context
	// from when we initialized.
	check_error();
	glDisable(GL_DITHER);
	check_error();
	glEnable(GL_FRAMEBUFFER_SRGB);
	check_error();

	// Save original viewport.
	GLuint x = 0, y = 0;

	if (width == 0 && height == 0) {
		GLint viewport[4];
		glGetIntegerv(GL_VIEWPORT, viewport);
		x = viewport[0];
		y = viewport[1];
		width = viewport[2];
		height = viewport[3];
	}

	// Basic state.
	check_error();
	glDisable(GL_BLEND);
	check_error();
	glDisable(GL_DEPTH_TEST);
	check_error();
	glDepthMask(GL_FALSE);
	check_error();

	// Generate a VAO that will be used during the entire execution,
	// and bind the VBO, since it contains all the data.
	GLuint vao;
	glGenVertexArrays(1, &vao);
	check_error();
	glBindVertexArray(vao);
	check_error();
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	check_error();
	set<GLint> bound_attribute_indices;

	set<Phase *> generated_mipmaps;

	// We choose the simplest option of having one texture per output,
	// since otherwise this turns into an (albeit simple) register allocation problem.
	map<Phase *, GLuint> output_textures;

	for (unsigned phase_num = 0; phase_num < phases.size(); ++phase_num) {
		Phase *phase = phases[phase_num];

		if (do_phase_timing) {
			GLuint timer_query_object;
			if (phase->timer_query_objects_free.empty()) {
				glGenQueries(1, &timer_query_object);
			} else {
				timer_query_object = phase->timer_query_objects_free.front();
				phase->timer_query_objects_free.pop_front();
			}
			glBeginQuery(GL_TIME_ELAPSED, timer_query_object);
			phase->timer_query_objects_running.push_back(timer_query_object);
		}
		if (phase_num == phases.size() - 1) {
			// Last phase goes to the output the user specified.
			glBindFramebuffer(GL_FRAMEBUFFER, dest_fbo);
			check_error();
			GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
			assert(status == GL_FRAMEBUFFER_COMPLETE);
			glViewport(x, y, width, height);
			if (dither_effect != NULL) {
				CHECK(dither_effect->set_int("output_width", width));
				CHECK(dither_effect->set_int("output_height", height));
			}
		}
		execute_phase(phase, phase_num == phases.size() - 1, &bound_attribute_indices, &output_textures, &generated_mipmaps);
		if (do_phase_timing) {
			glEndQuery(GL_TIME_ELAPSED);
		}
	}

	for (map<Phase *, GLuint>::const_iterator texture_it = output_textures.begin();
	     texture_it != output_textures.end();
	     ++texture_it) {
		resource_pool->release_2d_texture(texture_it->second);
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	check_error();
	glUseProgram(0);
	check_error();

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	check_error();
	glBindVertexArray(0);
	check_error();
	glDeleteVertexArrays(1, &vao);
	check_error();

	if (do_phase_timing) {
		// Get back the timer queries.
		for (unsigned phase_num = 0; phase_num < phases.size(); ++phase_num) {
			Phase *phase = phases[phase_num];
			for (std::list<GLuint>::iterator timer_it = phase->timer_query_objects_running.begin();
			     timer_it != phase->timer_query_objects_running.end(); ) {
				GLint timer_query_object = *timer_it;
				GLint available;
				glGetQueryObjectiv(timer_query_object, GL_QUERY_RESULT_AVAILABLE, &available);
				if (available) {
					GLuint64 time_elapsed;
					glGetQueryObjectui64v(timer_query_object, GL_QUERY_RESULT, &time_elapsed);
					phase->time_elapsed_ns += time_elapsed;
					++phase->num_measured_iterations;
					phase->timer_query_objects_free.push_back(timer_query_object);
					phase->timer_query_objects_running.erase(timer_it++);
				} else {
					++timer_it;
				}
			}
		}
	}
}

void EffectChain::enable_phase_timing(bool enable)
{
	if (enable) {
		assert(movit_timer_queries_supported);
	}
	this->do_phase_timing = enable;
}

void EffectChain::reset_phase_timing()
{
	for (unsigned phase_num = 0; phase_num < phases.size(); ++phase_num) {
		Phase *phase = phases[phase_num];
		phase->time_elapsed_ns = 0;
		phase->num_measured_iterations = 0;
	}
}

void EffectChain::print_phase_timing()
{
	double total_time_ms = 0.0;
	for (unsigned phase_num = 0; phase_num < phases.size(); ++phase_num) {
		Phase *phase = phases[phase_num];
		double avg_time_ms = phase->time_elapsed_ns * 1e-6 / phase->num_measured_iterations;
		printf("Phase %d: %5.1f ms  [", phase_num, avg_time_ms);
		for (unsigned effect_num = 0; effect_num < phase->effects.size(); ++effect_num) {
			if (effect_num != 0) {
				printf(", ");
			}
			printf("%s", phase->effects[effect_num]->effect->effect_type_id().c_str());
		}
		printf("]\n");
		total_time_ms += avg_time_ms;
	}
	printf("Total:   %5.1f ms\n", total_time_ms);
}

void EffectChain::execute_phase(Phase *phase, bool last_phase,
                                set<GLint> *bound_attribute_indices,
                                map<Phase *, GLuint> *output_textures,
                                set<Phase *> *generated_mipmaps)
{
	GLuint fbo = 0;

	// Find a texture for this phase.
	inform_input_sizes(phase);
	if (!last_phase) {
		find_output_size(phase);

		GLuint tex_num = resource_pool->create_2d_texture(intermediate_format, phase->output_width, phase->output_height);
		output_textures->insert(make_pair(phase, tex_num));
	}

	glUseProgram(phase->glsl_program_num);
	check_error();

	// Set up RTT inputs for this phase.
	for (unsigned sampler = 0; sampler < phase->inputs.size(); ++sampler) {
		glActiveTexture(GL_TEXTURE0 + sampler);
		Phase *input = phase->inputs[sampler];
		input->output_node->bound_sampler_num = sampler;
		glBindTexture(GL_TEXTURE_2D, (*output_textures)[input]);
		check_error();
		if (phase->input_needs_mipmaps && generated_mipmaps->count(input) == 0) {
			glGenerateMipmap(GL_TEXTURE_2D);
			check_error();
			generated_mipmaps->insert(input);
		}
		setup_rtt_sampler(sampler, phase->input_needs_mipmaps);
		phase->input_samplers[sampler] = sampler;  // Bind the sampler to the right uniform.
	}

	// And now the output. (Already set up for us if it is the last phase.)
	if (!last_phase) {
		fbo = resource_pool->create_fbo((*output_textures)[phase]);
		glBindFramebuffer(GL_FRAMEBUFFER, fbo);
		glViewport(0, 0, phase->output_width, phase->output_height);
	}

	// Give the required parameters to all the effects.
	unsigned sampler_num = phase->inputs.size();
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		unsigned old_sampler_num = sampler_num;
		node->effect->set_gl_state(phase->glsl_program_num, phase->effect_ids[node], &sampler_num);
		check_error();

		if (node->effect->is_single_texture()) {
			assert(sampler_num - old_sampler_num == 1);
			node->bound_sampler_num = old_sampler_num;
		} else {
			node->bound_sampler_num = -1;
		}
	}

	// Uniforms need to come after set_gl_state(), since they can be updated
	// from there.
	setup_uniforms(phase);

	// Clean up old attributes if they are no longer needed.
	for (set<GLint>::iterator attr_it = bound_attribute_indices->begin();
	     attr_it != bound_attribute_indices->end(); ) {
		if (phase->attribute_indexes.count(*attr_it) == 0) {
			glDisableVertexAttribArray(*attr_it);
			check_error();
			bound_attribute_indices->erase(attr_it++);
		} else {
			++attr_it;
		}
	}

	// Set up the new attributes, if needed.
	for (set<GLint>::iterator attr_it = phase->attribute_indexes.begin();
	     attr_it != phase->attribute_indexes.end();
	     ++attr_it) {
		if (bound_attribute_indices->count(*attr_it) == 0) {
			glEnableVertexAttribArray(*attr_it);
			check_error();
			glVertexAttribPointer(*attr_it, 2, GL_FLOAT, GL_FALSE, 0, BUFFER_OFFSET(0));
			check_error();
			bound_attribute_indices->insert(*attr_it);
		}
	}

	glDrawArrays(GL_TRIANGLES, 0, 3);
	check_error();
	
	for (unsigned i = 0; i < phase->effects.size(); ++i) {
		Node *node = phase->effects[i];
		node->effect->clear_gl_state();
	}

	if (!last_phase) {
		resource_pool->release_fbo(fbo);
	}
}

void EffectChain::setup_uniforms(Phase *phase)
{
	char *ubo_data = phase->ubo_data.empty() ? NULL : &phase->ubo_data[0];

	for (size_t i = 0; i < phase->uniforms_sampler2d.size(); ++i) {
		const Uniform<int> &uniform = phase->uniforms_sampler2d[i];
		if (uniform.location != GL_INVALID_INDEX) {
			glUniform1iv(uniform.location, uniform.num_values, uniform.value);
		}
		assert(uniform.ubo_offset == -1);  // Samplers don't go into UBOs.
	}
	for (size_t i = 0; i < phase->uniforms_bool.size(); ++i) {
		const Uniform<bool> &uniform = phase->uniforms_bool[i];
		assert(uniform.num_values == 1);
		if (uniform.location != GL_INVALID_INDEX) {
			glUniform1i(uniform.location, *uniform.value);
		}
		if (uniform.ubo_offset != -1) {
			GLint int_val = *uniform.value;
			memcpy(ubo_data + uniform.ubo_offset, &int_val, sizeof(int_val));
		}
	}
	for (size_t i = 0; i < phase->uniforms_int.size(); ++i) {
		const Uniform<int> &uniform = phase->uniforms_int[i];
		if (uniform.location != GL_INVALID_INDEX) {
			glUniform1iv(uniform.location, uniform.num_values, uniform.value);
		}
		if (uniform.ubo_offset != -1) {
			memcpy(ubo_data + uniform.ubo_offset, uniform.value, uniform.ubo_num_elem * sizeof(*uniform.value));
		}
	}
	for (size_t i = 0; i < phase->uniforms_float.size(); ++i) {
		const Uniform<float> &uniform = phase->uniforms_float[i];
		if (uniform.location != GL_INVALID_INDEX) {
			glUniform1fv(uniform.location, uniform.num_values, uniform.value);
		}
		if (uniform.ubo_offset != -1) {
			memcpy(ubo_data + uniform.ubo_offset, uniform.value, uniform.ubo_num_elem * sizeof(*uniform.value));
		}
	}
	for (size_t i = 0; i < phase->uniforms_vec2.size(); ++i) {
		const Uniform<float> &uniform = phase->uniforms_vec2[i];
		if (uniform.location != GL_INVALID_INDEX) {
			glUniform2fv(uniform.location, uniform.ubo_num_elem, uniform.value);
		}
		if (uniform.ubo_offset != -1) {
			memcpy(ubo_data + uniform.ubo_offset, uniform.value, uniform.ubo_num_elem * 2 * sizeof(*uniform.value));
		}
	}
	for (size_t i = 0; i < phase->uniforms_vec3.size(); ++i) {
		const Uniform<float> &uniform = phase->uniforms_vec3[i];
		if (uniform.location != GL_INVALID_INDEX) {
			glUniform3fv(uniform.location, uniform.ubo_num_elem, uniform.value);
		}
		if (uniform.ubo_offset != -1) {
			memcpy(ubo_data + uniform.ubo_offset, uniform.value, uniform.ubo_num_elem * 3 * sizeof(*uniform.value));
		}
	}
	for (size_t i = 0; i < phase->uniforms_vec4.size(); ++i) {
		const Uniform<float> &uniform = phase->uniforms_vec4[i];
		if (uniform.location != GL_INVALID_INDEX) {
			glUniform4fv(uniform.location, uniform.ubo_num_elem, uniform.value);
		}
		if (uniform.ubo_offset != -1) {
			memcpy(ubo_data + uniform.ubo_offset, uniform.value, uniform.ubo_num_elem * 4 * sizeof(*uniform.value));
		}
	}
	for (size_t i = 0; i < phase->uniforms_mat3.size(); ++i) {
		const Uniform<Matrix3d> &uniform = phase->uniforms_mat3[i];
		assert(uniform.ubo_num_elem == 1);
		if (uniform.location != GL_INVALID_INDEX) {
			// Convert to float (GLSL has no double matrices).
		        float matrixf[9];
			for (unsigned y = 0; y < 3; ++y) {
				for (unsigned x = 0; x < 3; ++x) {
					matrixf[y + x * 3] = (*uniform.value)(y, x);
				}
			}
			glUniformMatrix3fv(uniform.location, 1, GL_FALSE, matrixf);
		}
		if (uniform.ubo_offset != -1) {
			// TODO
			assert(false);
		}
	}

	if (phase->ubo != GL_INVALID_INDEX) {
		// TODO: Do we want to demand DSA for this?
		glNamedBufferSubData(phase->ubo, 0, phase->ubo_data.size(), ubo_data);
		return;
	}
}

void EffectChain::setup_rtt_sampler(int sampler_num, bool use_mipmaps)
{
	glActiveTexture(GL_TEXTURE0 + sampler_num);
	check_error();
	if (use_mipmaps) {
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);
		check_error();
	} else {
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		check_error();
	}
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	check_error();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	check_error();
}

}  // namespace movit
