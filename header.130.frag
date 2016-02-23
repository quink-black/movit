#version 130

// TODO: Make dependent on whether we actually use the feature,
// and also put in all the headers.
#extension GL_ARB_uniform_buffer_object : enable

in vec2 tc;

vec4 tex2D(sampler2D s, vec2 coord)
{
	return texture(s, coord);
}
