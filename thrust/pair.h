/*
 *  Copyright 2008-2013 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file pair.h
 *  \brief A type encapsulating a heterogeneous pair of elements
 */

#pragma once

#include <thrust/detail/config.h>

#include <cuda/std/utility>

namespace std
{
template <class T, class U>
class pair;
}

THRUST_NAMESPACE_BEGIN

/*! \addtogroup utility
 *  \{
 */

/*! \addtogroup pair
 *  \{
 */

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns either the type of a \p pair's
 *  \c first_type or \c second_type in its nested type, \c type.
 *
 *  \tparam N This parameter selects the member of interest.
 *  \tparam T A \c pair type of interest.
 */
template <size_t N, class T>
using tuple_element = ::cuda::std::tuple_element<N, T>;

/*! This convenience metafunction is included for compatibility with
 *  \p tuple. It returns \c 2, the number of elements of a \p pair,
 *  in its nested data member, \c value.
 *
 *  \tparam Pair A \c pair type of interest.
 */
template <class T>
using tuple_size = ::cuda::std::tuple_size<T>;

/*! \p pair is a generic data structure encapsulating a heterogeneous
 *  pair of values.
 *
 *  \tparam T1 The type of \p pair's first object type.  There are no
 *          requirements on the type of \p T1. <tt>T1</tt>'s type is
 *          provided by <tt>pair::first_type</tt>.
 *
 *  \tparam T2 The type of \p pair's second object type.  There are no
 *          requirements on the type of \p T2. <tt>T2</tt>'s type is
 *          provided by <tt>pair::second_type</tt>.
 */
template <class T, class U>
class pair : public ::cuda::std::pair<T, U>
{
public:
  using super_t = ::cuda::std::pair<T, U>;
  using super_t::super_t;

  // allow construction from std::pair
  template <class T2, class U2>
  inline THRUST_HOST_DEVICE
  pair(const ::std::pair<T2, U2>& other)
      : super_t(other.first, other.second)
  {}

  // allow assignment from std::pair
  template <class T2, class U2>
  inline THRUST_HOST_DEVICE
  pair& operator=(const ::std::pair<T2, U2>& other)
  {
    this->first  = other.first;
    this->second = other.second;
    return *this;
  }
};

// We cannot derive from `cuda::std::pair` directly, so we need to specialize `get`
template <size_t N, class T, class U>
inline THRUST_HOST_DEVICE
typename tuple_element<N, pair<T, U>>::type& get(pair<T, U>& t) noexcept
{
  using ::cuda::std::get;
  return get<N>(t);
}

template <size_t N, class T, class U>
inline THRUST_HOST_DEVICE
typename tuple_element<N, pair<T, U>>::type&& get(pair<T, U>&& t) noexcept
{
  using ::cuda::std::get;
  return get<N>(static_cast<pair<T, U>&&>(t));
}

template <size_t N, class T, class U>
inline THRUST_HOST_DEVICE
const typename tuple_element<N, pair<T, U>>::type& get(const pair<T, U>& t) noexcept
{
  using ::cuda::std::get;
  return get<N>(t);
}

template <size_t N, class T, class U>
inline THRUST_HOST_DEVICE
const typename tuple_element<N, pair<T, U>>::type&& get(const pair<T, U>&& t) noexcept
{
  using ::cuda::std::get;
  return get<N>(static_cast<const pair<T, U>&&>(t));
}

namespace detail
{
template <class T>
struct unwrap_refwrapper
{
  using type = T;
};

template <class T>
struct unwrap_refwrapper<std::reference_wrapper<T>>
{
  using type = T&;
};

template <class T>
using unwrap_decay_t = typename unwrap_refwrapper<typename ::cuda::std::decay<T>::type>::type;
} // namespace detail

/*! This convenience function creates a \p pair from two objects.
 *
 *  \param x The first object to copy from.
 *  \param y The second object to copy from.
 *  \return A newly-constructed \p pair copied from \p a and \p b.
 *
 *  \tparam T1 There are no requirements on the type of \p T1.
 *  \tparam T2 There are no requirements on the type of \p T2.
 */
template <class T, class U>
inline THRUST_HOST_DEVICE
pair<detail::unwrap_decay_t<T>, detail::unwrap_decay_t<U>> make_pair(T&& first, U&& second)
{
  return {::cuda::std::forward<T>(first), ::cuda::std::forward<U>(second)};
}

/*! \endcond
 */

/*! \} // tuple
 */

/*! \} // utility
 */

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// define tuple_size, tuple_element, etc.
template <class T, class U>
struct tuple_size<THRUST_NS_QUALIFIER::pair<T, U>> : std::integral_constant<size_t, 2>
{};

template <class T, class U>
struct tuple_element<0, THRUST_NS_QUALIFIER::pair<T, U>>
{
  using type = T;
};

template <class T, class U>
struct tuple_element<1, THRUST_NS_QUALIFIER::pair<T, U>>
{
  using type = U;
};

_LIBCUDACXX_END_NAMESPACE_STD
