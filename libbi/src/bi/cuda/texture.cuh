/**
 * @file
 *
 * Functions for efficient reading of state objects through the texture
 * cache.
 *
 * @author Lawrence Murray <lawrence.murray@csiro.au>
 * $Rev$
 * $Date$
 *
 * Some of these functions rely on constant memory also and should be used in
 * conjunction with those in constant.cuh. First bind a model object to the
 * global constant memory variables using constant_bind() on the host, then
 * bind a state object to texture using texture_bind(). Any of the fetch
 * functions may then be used on the device to read from the state via the
 * texture cache. When finished, call texture_unbind() on the host to free
 * textures for reuse.
 */
#ifndef BI_CUDA_TEXTURE_CUH
#define BI_CUDA_TEXTURE_CUH

#include "cuda.hpp"

/**
 * @internal
 *
 * @def TEXTURE_STATE_DEC
 *
 * Macro for state texture reference declarations.
 *
 * Note textures do not support doubles, and so we explicitly use float here
 * rather than real.
 */
#define TEXTURE_STATE_DEC(Name) \
  /**
   @internal

   Global model texture reference.
   */ \
  texture<float,2,cudaReadModeElementType> tex##Name##State;

TEXTURE_STATE_DEC(D)
TEXTURE_STATE_DEC(DX)
TEXTURE_STATE_DEC(R)
TEXTURE_STATE_DEC(F)
TEXTURE_STATE_DEC(O)
TEXTURE_STATE_DEC(P)
TEXTURE_STATE_DEC(PX)

namespace bi {
/**
 * @internal
 *
 * @def TEXTURE_BIND_DEC
 *
 * Macro for texture bind function declarations.
 */
#define TEXTURE_BIND_DEC(name) \
  /**
    @internal

    Bind name##-net state to texture reference.

    @ingroup state_gpu

    @param s State.
   */ \
  template<class M1> \
  CUDA_FUNC_HOST void texture_bind_##name(M1& s);

TEXTURE_BIND_DEC(d)
TEXTURE_BIND_DEC(dx)
TEXTURE_BIND_DEC(r)
TEXTURE_BIND_DEC(f)
TEXTURE_BIND_DEC(o)
TEXTURE_BIND_DEC(p)
TEXTURE_BIND_DEC(px)

/**
 * @internal
 *
 * @def TEXTURE_UNBIND_DEC
 *
 * Macro for texture unbind function declarations.
 */
#define TEXTURE_UNBIND_DEC(name) \
  /**
    @internal

    Unbind state from global textures.

    @ingroup state_gpu
   */ \
  CUDA_FUNC_HOST void texture_unbind_##name();

TEXTURE_UNBIND_DEC(d)
TEXTURE_UNBIND_DEC(dx)
TEXTURE_UNBIND_DEC(r)
TEXTURE_UNBIND_DEC(f)
TEXTURE_UNBIND_DEC(o)
TEXTURE_UNBIND_DEC(p)
TEXTURE_UNBIND_DEC(px)

/**
 * Texture-assisted fetch of name##-var value from global memory.
 *
 * @ingroup state_gpu
 *
 * @tparam B Model type.
 * @tparam X Node type.
 *
 * @param p Trajectory id.
 * @param cox Serial coordinate.
 *
 * @return Value of the given node for the trajectory associated with the
 * calling thread.
 */
template<class B, class X>
CUDA_FUNC_DEVICE float texture_fetch(const int p, const int cox);

/**
 * Facade for state in texture memory.
 *
 * @ingroup state_gpu
 */
struct texture {
  static const bool on_device = true;

  /**
   * Fetch.
   */
  template<class B, class X>
  static CUDA_FUNC_DEVICE float fetch(const int p, const int cox) {
    return texture_fetch<B,X>(p, cox);
  }
};

}

#include "constant.cuh"

/**
 * @def TEXTURE_BIND_DEF
 *
 * Macro for texture bind function definintions.
 *
 * @note As of CUDA 2.3, these appear to be needed in the same compilation
 * unit (*.cu file) as where the textures are declared.
 */
#define TEXTURE_BIND_DEF(name, Name) \
  template<class M1> \
  inline void bi::texture_bind_##name(M1& s) { \
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>(); \
    CUDA_CHECKED_CALL(cudaBindTexture2D(0, &tex##Name##State, s.buf(), \
        &channelDesc, s.size1(), s.size2(), s.lead()*sizeof(real))); \
  }

TEXTURE_BIND_DEF(d, D)
TEXTURE_BIND_DEF(dx, DX)
TEXTURE_BIND_DEF(r, R)
TEXTURE_BIND_DEF(f, F)
TEXTURE_BIND_DEF(o, O)
TEXTURE_BIND_DEF(p, P)
TEXTURE_BIND_DEF(px, PX)

/**
 * @def TEXTURE_UNBIND_DEF
 *
 * Macro for texture unbind function definitions.
 *
 * @note As of CUDA 2.3, these appear to be needed in the same compilation
 * unit (*.cu file) as where the textures are declared.
 */
#define TEXTURE_UNBIND_DEF(name, Name) \
  inline void bi::texture_unbind_##name() { \
    CUDA_CHECKED_CALL(cudaUnbindTexture(tex##Name##State)); \
  }

TEXTURE_UNBIND_DEF(d, D)
TEXTURE_UNBIND_DEF(dx, DX)
TEXTURE_UNBIND_DEF(r, R)
TEXTURE_UNBIND_DEF(f, F)
TEXTURE_UNBIND_DEF(o, O)
TEXTURE_UNBIND_DEF(p, P)
TEXTURE_UNBIND_DEF(px, PX)

template<class B, class X>
inline float bi::texture_fetch(const int p, const int cox) {
  const int i = var_net_start<B,X>::value + cox;

  if (is_d_var<X>::value) {
    return tex2D(texDState, p, i);
  } else if (is_dx_var<X>::value) {
    return tex2D(texDXState, p, i);
  } else if (is_r_var<X>::value) {
    return tex2D(texRState, p, i);
  } else if (is_f_var<X>::value) {
    return tex2D(texFState, 0, i);
  } else if (is_o_var<X>::value) {
    return tex2D(texOState, p, i);
  } else if (is_p_var<X>::value) {
    return tex2D(texPState, 0, i);
  } else if (is_px_var<X>::value) {
    return tex2D(texPXState, 0, i);
  } else {
    return BI_REAL(1.0 / 0.0);
  }
}

#endif
