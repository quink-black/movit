#ifndef _WHITE_BALANCE_EFFECT_H
#define _WHITE_BALANCE_EFFECT_H 1

// Color correction in LMS color space.

#include "effect.h"

class WhiteBalanceEffect : public Effect {
public:
	WhiteBalanceEffect();
	virtual std::string effect_type_id() const { return "WhiteBalanceEffect"; }
	std::string output_fragment_shader();

	void set_gl_state(GLuint glsl_program_num, const std::string &prefix, unsigned *sampler_num);

private:
	// The neutral color, in linear sRGB.
	RGBTriplet neutral_color;

	// Output color temperature (in Kelvins).
	// Choosing 6500 will lead to no color cast (ie., the neutral color becomes perfectly gray).
	float output_color_temperature;
};

#endif // !defined(_WHITE_BALANCE_EFFECT_H)