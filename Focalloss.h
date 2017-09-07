// Focalloss.h
#ifndef FOCALLOSS_H_
#define FOCALLOSS_H_

template <typename Device, typename T>
struct FOCALLOSSFunctor {
  void operator()(const Device& d, const T* in, T* out);
};

#endif FOCALLOSS_H_
