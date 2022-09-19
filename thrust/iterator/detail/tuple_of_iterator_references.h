/*
 *  Copyright 2008-2018 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>
#include <thrust/detail/reference_forward_declaration.h>
#include <thrust/pair.h>
#include <thrust/tuple.h>

THRUST_NAMESPACE_BEGIN
namespace detail
{

template <typename... Ts>
class tuple_of_iterator_references : public thrust::tuple<Ts...>
{
  using super_t = thrust::tuple<Ts...>;
  using super_t::super_t;

public:
  // allow implicit construction from tuple<refs>
  inline THRUST_HOST_DEVICE
  tuple_of_iterator_references(const super_t& other)
      : super_t(other)
  {}

  // allow assignment from tuples
  __thrust_exec_check_disable__
  template <typename... Us>
  inline THRUST_HOST_DEVICE
  tuple_of_iterator_references& operator=(const thrust::tuple<Us...>& other)
  {
    super_t::operator=(other);
    return *this;
  }

  // allow assignment from pairs
  __thrust_exec_check_disable__
  template <typename U1, typename U2>
  inline THRUST_HOST_DEVICE
  tuple_of_iterator_references& operator=(const thrust::pair<U1, U2>& other)
  {
    get<0>(*this) = other.first;
    get<1>(*this) = other.second;
    return *this;
  }

  // allow assignment from reference<tuple>
  // XXX perhaps we should generalize to reference<T>
  //     we could captures reference<pair> this way
  __thrust_exec_check_disable__
  template <typename Pointer, typename Derived, typename... Us>
  inline THRUST_HOST_DEVICE
  tuple_of_iterator_references&
  operator=(const thrust::reference<thrust::tuple<Us...>, Pointer, Derived>& other)
  {
    typedef thrust::tuple<Us...> tuple_type;

    // XXX perhaps this could be accelerated
    tuple_type other_tuple = other;
    super_t::operator=(other_tuple);
    return *this;
  }

  // allow conversion to tuple
  // XXX perhaps we should constraint with enable_if
  template <class... Us>
  inline THRUST_HOST_DEVICE
  operator thrust::tuple<Us...>() const { return thrust::tuple<Us...>{*this}; }
};

} // namespace detail

THRUST_NAMESPACE_END

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// define tuple_size, tuple_element, etc.
template <class... Ts>
struct tuple_size<THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>
    : std::integral_constant<size_t, sizeof...(Ts)>
{};

template <size_t i>
struct tuple_element<i, THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<>>
{};

template <class T, class... Ts>
struct tuple_element<0, THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<T, Ts...>>
{
  using type = T;
};

template <size_t i, class T, class... Ts>
struct tuple_element<i, THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<T, Ts...>>
{
  using type =
    typename tuple_element<i - 1,
                           THRUST_NS_QUALIFIER::detail::tuple_of_iterator_references<Ts...>>::type;
};

_LIBCUDACXX_END_NAMESPACE_STD
