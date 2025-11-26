#include "common_utils.h"
#include <cmath>
#include <cstdlib>

float wrapToPi(float x) {
    const float pi = static_cast<float>(M_PI);
    const float twoPi = 2.0f * pi;
    x = fmodf(x + pi, twoPi);
    if (x < 0.0f)
        x += twoPi;
    return x - pi;
}
