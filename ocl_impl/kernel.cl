
#define SCALE 0.65f

__constant sampler_t volume_sampler = CLK_NORMALIZED_COORDS_FALSE |
                                        CLK_ADDRESS_CLAMP_TO_EDGE |
                                        CLK_FILTER_LINEAR;

float4 transfer_function(float x) {
    return (float4) {
        1.0f * exp( -pow(x - 9.0f, 2.0f) / 1.0f ) + 0.1f * exp( -pow(x - 3.0f, 2.0f) / 0.1f ) + 0.1f  * exp( -pow(x - -3.0f, 2.0f) / 0.5f ),
        1.0f * exp( -pow(x - 9.0f, 2.0f) / 1.0f ) + 1.0f * exp( -pow(x - 3.0f, 2.0f) / 0.1f ) + 0.1f  * exp( -pow(x - -3.0f, 2.0f) / 0.5f ),
        0.1f * exp( -pow(x - 9.0f, 2.0f) / 1.0f ) + 0.1f * exp( -pow(x - 3.0f, 2.0f) / 0.1f ) + 1.0f  * exp( -pow(x - -3.0f, 2.0f) / 0.5f ),
        0.6f * exp( -pow(x - 9.0f, 2.0f) / 1.0f ) + 0.1f * exp( -pow(x - 3.0f, 2.0f) / 0.1f ) + 0.01f * exp( -pow(x - -3.0f, 2.0f) / 0.5f ),
    };
}

__kernel void render(__read_only image3d_t volume_image,
                     __write_only image2d_t result_image,
                     float angle,
                     int3 res,
                     int3 dims) {
    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);

    float4 sum = { 0, 0, 0, 0 };
    for (int iz = res[2]-1; iz >= 0; iz--) {

        float x = ((ix / (res[0] - 1.0f)) - 0.5f) * (dims[0] * SCALE);
        float y = ((iy / (res[1] - 1.0f)) - 0.5f) * (dims[1] * SCALE);
        float z = ((iz / (res[2] - 1.0f)) - 0.5f) * (dims[2] * SCALE);

        float fx = x * cos(angle) - z * sin(angle);
        float fy = y;
        float fz = x * sin(angle) + z * cos(angle);

        fx += (dims[0]/2);
        fy += (dims[1]/2);
        fz += (dims[2]/2);

        float density = read_imagef(volume_image, volume_sampler, (float4){ fx, fy, fz, 0 }).r;

        float4 c = transfer_function(log(density));

        sum.r = c.a * c.r + (1 - c.a) * sum.r;
        sum.g = c.a * c.g + (1 - c.a) * sum.g;
        sum.b = c.a * c.b + (1 - c.a) * sum.b;
    }

    sum.a = 1;
    write_imagef(result_image, (int2){ix, iy}, sum);
}
