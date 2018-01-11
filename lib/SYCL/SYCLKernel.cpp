//===- SYCLKernel.cpp                                       ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect SYCL kernels and rename kernel to shorten unique names
//
// Detect if functions have kernel as an ancestor
// ===------------------------------------------------------------------- -===//

#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/CallGraphSCCPass.h"
// Wait for LLVM 4.0...
// #include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/CallSite.h"
#include "llvm/SYCL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

namespace llvm {
namespace sycl {

/// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
/// option -debug or -debug-only=SYCL
#define DEBUG_TYPE "SYCL"

/// The function template used to instantiate a kernel inside triSYCL
/// is used as marker to detect the kernel functions
StringRef SYCLKernelPrefix { "void cl::sycl::detail::instantiate_kernel<" };

/// For some implementations, rename the kernels with shorter and cleaner names
/// starting with this prefix
std::string SYCLKernelShortPrefix { "TRISYCL_kernel_" };

/// Test if a function is a SYCL kernel
bool isKernel(const Function &F) {
  auto lookForKernel = [&] {
    if (!F.hasName())
      return false;

    if (F.getName().startswith(SYCLKernelShortPrefix))
      return true;

    // Demangle C++ name for human beings
    int Status;
    // Use a \c unique_ptr to deallocate memory on exit
    std::unique_ptr<const char> Demangled {
      itaniumDemangle(F.getName().str().c_str(),
                      nullptr,
                      nullptr,
                      &Status) };
    if (Demangled) {
      DEBUG(errs() << " Demangled: " << Demangled.get() << "\n");
      // A kernel is just a function starting with the well known name
      return StringRef { Demangled.get() }.startswith(SYCLKernelPrefix);
    }
    return false;
  };

  auto KernelFound = lookForKernel();
  DEBUG(if (KernelFound)
          errs() << "\n\tKernel found!\n\n");

  return KernelFound;
}

/// Test if a function is in the list of having kernel ancestor
bool isInHasKernelAncestorFunctionList(Function &F,
                       std::vector<Function *> &hasKernelAncestorFunctionList) {
  if (std::find(hasKernelAncestorFunctionList.begin(), hasKernelAncestorFunctionList.end(), &F)
      != hasKernelAncestorFunctionList.end())
    return true;
  else
    return false;
}

/// Test if a function has ancestor kernel
bool hasAncestorKernel(Function &F,
                       std::vector<Function *> &hasKernelAncestorFunctionList) {
  for (auto &U : F.uses()) {
    CallSite CS{U.getUser()};
    if (auto I = CS.getInstruction()) {
      auto parent = I->getParent()->getParent();
      if (isInHasKernelAncestorFunctionList(*parent, hasKernelAncestorFunctionList))
        return true;
      else
        return false;
    }
  }

  return false;
}

/// Compute CallGraph to record all functions that have ancestor kernel
void computeAncestorNode(CallGraphSCC &SCC, CallGraph &CG,
                         std::vector<Function *> &hasKernelAncestorFunctionList) {
  // Find the CGN the kernel function belongs to
  // DFS algorithm start from the kernel function CGN to record all the function
  // has ancestor of it
  // for (CallGraphNode *I : SCC) {} will run in bottom-up order
  for (auto SCCI = scc_begin(&CG); !SCCI.isAtEnd();
       ++SCCI) {
    const std::vector<CallGraphNode*> &nextSCC = *SCCI;
    for (auto I = nextSCC.begin(), E = nextSCC.end(); I != E; ++I) {
      auto *F = (*I)->getFunction();
      if(F)
        if (isKernel(*F) || hasAncestorKernel(*F, hasKernelAncestorFunctionList))
          hasKernelAncestorFunctionList.push_back(F);
    }
  }
}

/// Update functions that have ancestor kernel list when new CallGraphNode created in CallGraph
void updateHasKernelAncestorFunctionList (CallGraphNode &NewNode,
                                          std::vector<Function *> &hasKernelAncestorFunctionList) {
  auto *F = NewNode.getFunction();
  if (hasAncestorKernel(*F, hasKernelAncestorFunctionList))
    hasKernelAncestorFunctionList.push_back(F);
}

/// Mapping from the full kernel mangle named to a unique integer ID
std::map<std::string, std::size_t> SimplerKernelNames;

/// The kernel ID to use, counting from 0
std::size_t NextKernelID = 0;


/// Register a kernel with its full name and returns its ID
///
/// If the kernel is already registered, do not register it again.
std::size_t registerSYCLKernel(const std::string &LongKernelName) {
  auto Translation = SimplerKernelNames.emplace(LongKernelName, NextKernelID);
  /// If a new kernel has been registered, then compute the next ID to use
  if (Translation.second)
    ++NextKernelID;
  // Return the ID for the kernel
  return Translation.first->second;
}


/// Construct a kernel short name for an ID
std::string constructSYCLKernelShortName(std::size_t Id) {
  std::stringstream S;
  S << SYCLKernelShortPrefix << Id;
  return S.str();
}


/// Register a kernel with its full name and returns its short name
///
/// If the kernel is already registered, do not register it again.
std::string
registerSYCLKernelAndGetShortName(const std::string &LongKernelName) {
  return constructSYCLKernelShortName(registerSYCLKernel(LongKernelName));
}

}
}
