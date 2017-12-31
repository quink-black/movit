#ifndef _MOVIT_RESAMPLE_EFFECT_H
#define _MOVIT_RESAMPLE_EFFECT_H 1

// High-quality image resizing, either up or down.
//
// The default scaling offered by the GPU (and as used in ResizeEffect)
// is bilinear (optionally mipmapped), which is not the highest-quality
// choice, especially for upscaling. ResampleEffect offers the three-lobed
// Lanczos kernel, which is among the most popular choices in image
// processing. While it does have its weaknesses, in particular a certain
// ringing/sharpening effect with artifacts that accumulate over several
// consecutive resizings, it is generally regarded as the best tradeoff.
//
// Works in two passes; first horizontal, then vertical (ResampleEffect,
// which is what the user is intended to use, instantiates two copies of
// SingleResamplePassEffect behind the scenes).

#include <epoxy/gl.h>
#include <assert.h>
#include <stddef.h>
#include <memory>
#include <string>

#include "effect.h"
#include "fp16.h"

namespace movit {

class EffectChain;
class Node;
class SingleResamplePassEffect;
class ResampleComputeEffect;

// Public so that it can be benchmarked externally.
template<class T>
struct Tap {
	T weight;
	T pos;
};
struct ScalingWeights {
	unsigned src_bilinear_samples;
	unsigned dst_samples, num_loops;
	int int_radius;  // FIXME: really here?
	float scaling_factor;  // FIXME: really here?

	// Exactly one of these three is set.
	std::unique_ptr<Tap<fp16_int_t>[]> bilinear_weights_fp16;
	std::unique_ptr<Tap<float>[]> bilinear_weights_fp32;
	std::unique_ptr<fp16_int_t[]> raw_weights;
};
enum class BilinearFormatConstraints {
	ALLOW_FP16_AND_FP32,
	ALLOW_FP32_ONLY
};
ScalingWeights calculate_bilinear_scaling_weights(unsigned src_size, unsigned dst_size, float zoom, float offset, BilinearFormatConstraints constraints);
ScalingWeights calculate_raw_scaling_weights(unsigned src_size, unsigned dst_size, float zoom, float offset);

// A simple manager for support data stored in a 2D texture.
// Consider moving it to a shared location of more classes
// should need similar functionality.
class Support2DTexture {
public:
	Support2DTexture();
	~Support2DTexture();

	void update(GLint width, GLint height, GLenum internal_format, GLenum format, GLenum type, const GLvoid * data);
	GLint get_texnum() const { return texnum; }

private:
	GLuint texnum = 0;
	GLint last_texture_width = -1, last_texture_height = -1;
	GLenum last_texture_internal_format = GL_INVALID_ENUM;
};

class ResampleEffect : public Effect {
public:
	ResampleEffect();
	~ResampleEffect();

	std::string effect_type_id() const override { return "ResampleEffect"; }

	void inform_input_size(unsigned input_num, unsigned width, unsigned height) override;

	std::string output_fragment_shader() override {
		assert(false);
	}
	void set_gl_state(GLuint glsl_program_num, const std::string &prefix, unsigned *sampler_num) override {
		assert(false);
	}

	void rewrite_graph(EffectChain *graph, Node *self) override;
	bool set_float(const std::string &key, float value) override;
	
private:
	void update_size();
	void update_offset_and_zoom();

	// If compute shaders are supported, contains the effect.
	// If not, nullptr.
	std::unique_ptr<ResampleComputeEffect> compute_effect_owner;
	ResampleComputeEffect *compute_effect = nullptr;
	
	// Both of these are owned by us if owns_effects is true (before finalize()),
	// and otherwise owned by the EffectChain.
	std::unique_ptr<SingleResamplePassEffect> hpass_owner, vpass_owner;
	SingleResamplePassEffect *hpass = nullptr, *vpass = nullptr;

	int input_width, input_height, output_width, output_height;

	float offset_x, offset_y;
	float zoom_x, zoom_y;
	float zoom_center_x, zoom_center_y;
};

class SingleResamplePassEffect : public Effect {
public:
	// If parent is non-nullptr, calls to inform_input_size will be forwarded,
	// so that it can inform both passes about the right input and output
	// resolutions.
	SingleResamplePassEffect(ResampleEffect *parent);
	~SingleResamplePassEffect();
	std::string effect_type_id() const override { return "SingleResamplePassEffect"; }

	std::string output_fragment_shader() override;

	bool needs_texture_bounce() const override { return true; }
	bool needs_srgb_primaries() const override { return false; }
	AlphaHandling alpha_handling() const override { return INPUT_PREMULTIPLIED_ALPHA_KEEP_BLANK; }

	// We specifically do not want mipmaps on the input texture;
	// they break minification.
	MipmapRequirements needs_mipmaps() const override { return CANNOT_ACCEPT_MIPMAPS; }

	void inform_added(EffectChain *chain) override { this->chain = chain; }
	void inform_input_size(unsigned input_num, unsigned width, unsigned height) override {
		if (parent != nullptr) {
			parent->inform_input_size(input_num, width, height);
		}
	}
	bool changes_output_size() const override { return true; }
	bool sets_virtual_output_size() const override { return false; }

	void get_output_size(unsigned *width, unsigned *height, unsigned *virtual_width, unsigned *virtual_height) const override {
		*virtual_width = *width = this->output_width;
		*virtual_height = *height = this->output_height;
	}

	void set_gl_state(GLuint glsl_program_num, const std::string &prefix, unsigned *sampler_num) override;
	
	enum Direction { HORIZONTAL = 0, VERTICAL = 1 };

private:
	void update_texture(GLuint glsl_program_num, const std::string &prefix, unsigned *sampler_num);

	ResampleEffect *parent;
	EffectChain *chain;
	Direction direction;
	GLint uniform_sample_tex;
	float uniform_num_loops, uniform_slice_height, uniform_sample_x_scale, uniform_sample_x_offset;
	float uniform_whole_pixel_offset;
	int uniform_num_samples;

	int input_width, input_height, output_width, output_height;
	float offset, zoom;
	int last_input_width, last_input_height, last_output_width, last_output_height;
	float last_offset, last_zoom;
	int src_bilinear_samples, num_loops;
	float slice_height;
	Support2DTexture tex;
};

class ResampleComputeEffect : public Effect {
public:
	// If parent is non-nullptr, calls to inform_input_size will be forwarded,
	// so that it can inform both passes about the right input and output
	// resolutions.
	ResampleComputeEffect(ResampleEffect *parent);
	~ResampleComputeEffect();
	std::string effect_type_id() const override { return "ResampleComputeEffect"; }

	std::string output_fragment_shader() override;

	// FIXME: This is the primary reason why this doesn't really work;
	// there's no good reason why the regular resize should have bounce
	// but we shouldn't. (If we did a 2D block instead of 1D columns,
	// it would have been different, but we can't, due to the large size
	// of the fringe.)
	bool needs_texture_bounce() const override { return false; }
	bool needs_srgb_primaries() const override { return false; }
	AlphaHandling alpha_handling() const override { return INPUT_PREMULTIPLIED_ALPHA_KEEP_BLANK; }

	// We specifically do not want mipmaps on the input texture;
	// they break minification.
	MipmapRequirements needs_mipmaps() const override { return CANNOT_ACCEPT_MIPMAPS; }

	void inform_added(EffectChain *chain) override { this->chain = chain; }
	void inform_input_size(unsigned input_num, unsigned width, unsigned height) override {
		if (parent != nullptr) {
			parent->inform_input_size(input_num, width, height);
		}
	}
	bool changes_output_size() const override { return true; }
	bool sets_virtual_output_size() const override { return false; }

	void get_output_size(unsigned *width, unsigned *height, unsigned *virtual_width, unsigned *virtual_height) const override {
		*virtual_width = *width = this->output_width;
		*virtual_height = *height = this->output_height;
	}

	bool is_compute_shader() const override { return true; }
	void get_compute_dimensions(unsigned output_width, unsigned output_height,
	                            unsigned *x, unsigned *y, unsigned *z) const override;

	void set_gl_state(GLuint glsl_program_num, const std::string &prefix, unsigned *sampler_num) override;

private:
	void update_texture(GLuint glsl_program_num, const std::string &prefix, unsigned *sampler_num);

	ResampleEffect *parent;
	EffectChain *chain;
	Support2DTexture tex_horiz, tex_vert;
	GLint uniform_sample_tex_horizontal, uniform_sample_tex_vertical;
	float uniform_num_x_loops;
	int uniform_num_horizontal_filters, uniform_num_vertical_filters;
	float uniform_slice_height;
	float uniform_horizontal_whole_pixel_offset;
	int uniform_vertical_whole_pixel_offset;
	int uniform_num_horizontal_samples, uniform_num_vertical_samples;
	int uniform_output_samples_per_block;

	int input_width, input_height, output_width, output_height;
	float offset_x, offset_y, zoom_x, zoom_y;
	int last_input_width, last_input_height, last_output_width, last_output_height;
	float last_offset_x, last_offset_y, last_zoom_x, last_zoom_y;
	int src_horizontal_bilinear_samples;  // Horizontal.
	int src_vertical_samples;
	float slice_height;
	float uniform_inv_input_height, uniform_input_texcoord_y_adjust;
	int uniform_vertical_int_radius;
	float vertical_scaling_factor;
	float uniform_inv_vertical_scaling_factor;
};

}  // namespace movit

#endif // !defined(_MOVIT_RESAMPLE_EFFECT_H)
