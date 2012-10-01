#ifndef _EFFECT_CHAIN_H
#define _EFFECT_CHAIN_H 1

#include <vector>

#include "effect.h"
#include "effect_id.h"

enum PixelFormat { FORMAT_RGB, FORMAT_RGBA };

enum ColorSpace {
	COLORSPACE_sRGB = 0,
	COLORSPACE_REC_709 = 0,  // Same as sRGB.
	COLORSPACE_REC_601_525 = 1,
	COLORSPACE_REC_601_625 = 2,
};

enum GammaCurve {
	GAMMA_LINEAR = 0,
	GAMMA_sRGB = 1,
	GAMMA_REC_601 = 2,
	GAMMA_REC_709 = 2,  // Same as Rec. 601.
};

struct ImageFormat {
	PixelFormat pixel_format;
	ColorSpace color_space;
	GammaCurve gamma_curve;
};

class EffectChain {
public:
	EffectChain(unsigned width, unsigned height);
	void add_input(const ImageFormat &format);

	// The pointer is owned by EffectChain.
	Effect *add_effect(EffectId effect);

	void add_output(const ImageFormat &format);

	void render(unsigned char *src, unsigned char *dst);

private:
	unsigned width, height;
	ImageFormat input_format, output_format;
	std::vector<Effect *> effects;

	ColorSpace current_color_space;
	GammaCurve current_gamma_curve;	
};


#endif // !defined(_EFFECT_CHAIN_H)