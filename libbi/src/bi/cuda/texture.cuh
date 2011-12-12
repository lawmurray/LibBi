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
#include "../state/Coord.hpp"

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

TEXTURE_STATE_DEC(S)
TEXTURE_STATE_DEC(D)
TEXTURE_STATE_DEC(C)
TEXTURE_STATE_DEC(R)
TEXTURE_STATE_DEC(F)
TEXTURE_STATE_DEC(O)
TEXTURE_STATE_DEC(P)
TEXTURE_STATE_DEC(OY)
TEXTURE_STATE_DEC(OR)

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

TEXTURE_BIND_DEC(s)
TEXTURE_BIND_DEC(d)
TEXTURE_BIND_DEC(c)
TEXTURE_BIND_DEC(r)
TEXTURE_BIND_DEC(f)
TEXTURE_BIND_DEC(o)
TEXTURE_BIND_DEC(p)
TEXTURE_BIND_DEC(oy)
TEXTURE_BIND_DEC(or)

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

TEXTURE_UNBIND_DEC(s)
TEXTURE_UNBIND_DEC(d)
TEXTURE_UNBIND_DEC(c)
TEXTURE_UNBIND_DEC(r)
TEXTURE_UNBIND_DEC(f)
TEXTURE_UNBIND_DEC(o)
TEXTURE_UNBIND_DEC(p)
TEXTURE_UNBIND_DEC(oy)
TEXTURE_UNBIND_DEC(or)

/**
 * @internal
 *
 * Texture-assisted fetch of name##-node value from global memory.
 *
 * @ingroup state_gpu
 *
 * @tparam B Model type.
 * @tparam X Node type.
 * @tparam Xo X-offset.
 * @tparam Yo Y-offset.
 * @tparam Zo Z-offset.
 *
 * @param p Trajectory id. Ignored for f- and oy-node requests, as only one
 * trajectory is ever stored.
 * @param cox Base coordinates.
 *
 * @return Value of the given node for the trajectory associated with the
 * calling thread.
 */
template<class B, class X, int Xo, int Yo, int Zo>
CUDA_FUNC_DEVICE float texture_fetch(const int p, const Coord& cox);

/**
 * @internal
 *
 * Facade for state in texture memory.
 *
 * @ingroup state_gpu
 */
struct texture {
  static const bool on_device = true;

  /**
   * Fetch.
   */
  template<class B, class X, int Xo, int Yo, int Zo>
  static CUDA_FUNC_DEVICE float fetch(const int p, const Coord& cox) {
    return texture_fetch<B,X,Xo,Yo,Zo>(p, cox);
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

TEXTURE_BIND_DEF(s, S)
TEXTURE_BIND_DEF(d, D)
TEXTURE_BIND_DEF(c, C)
TEXTURE_BIND_DEF(r, R)
TEXTURE_BIND_DEF(f, F)
TEXTURE_BIND_DEF(o, O)
TEXTURE_BIND_DEF(p, P)
TEXTURE_BIND_DEF(oy, OY)
TEXTURE_BIND_DEF(or, OR)

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

TEXTURE_UNBIND_DEF(s, S)
TEXTURE_UNBIND_DEF(d, D)
TEXTURE_UNBIND_DEF(c, C)
TEXTURE_UNBIND_DEF(r, R)
TEXTURE_UNBIND_DEF(f, F)
TEXTURE_UNBIND_DEF(o, O)
TEXTURE_UNBIND_DEF(p, P)
TEXTURE_UNBIND_DEF(oy, OY)
TEXTURE_UNBIND_DEF(or, OR)

template<class B, class X, int Xo, int Yo, int Zo>
inline float bi::texture_fetch(const int p, const Coord& cox) {
  const int i = cox.Coord::index<B,X,Xo,Yo,Zo>();

  if (is_s_node<X>::value) {
    return tex2D(texSState, p, i);
  } else if (is_d_node<X>::value) {
    return tex2D(texDState, p, i);
  } else if (is_c_node<X>::value) {
    return tex2D(texCState, p, i);
  } else if (is_r_node<X>::value) {
    return tex2D(texRState, p, i);
  } else if (is_f_node<X>::value) {
    return tex2D(texFState, 0, i);
  } else if (is_o_node<X>::value) {
    return tex2D(texOState, 0, i);
  } else if (is_p_node<X>::value) {
    return tex2D(texPState, p, i);
  } else {
    return REAL(1.0 / 0.0);
  }
}

#endif
