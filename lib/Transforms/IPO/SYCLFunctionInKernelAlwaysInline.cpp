//===- SYCLFunctionInKernelAlwaysInline.cpp                               ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect functions that did not specify with noinline attribute called in SYCL 
// kernel. And force these functions inline.
// This pass needs to be done before inline pass and is aim to solve argument 
// flattening problem in SYCL.
// ===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/CallSite.h"
#include "llvm/SYCL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CtorUtils.h"

using namespace llvm;


/// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
/// option -debug or -debug-only=SYCL
#define DEBUG_TYPE "SYCL"

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Detect functions that did not specify with noinline attribute called in SYCL
/// kernel. And force these functions inline.
struct SYCLFunctionInKernelAlwaysInline : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  explicit SYCLFunctionInKernelAlwaysInline() : ModulePass(ID) {
     initializeSYCLFunctionInKernelAlwaysInlinePass(*PassRegistry::getPassRegistry());
  }


  bool doInitialization(Module &M) override {
    DEBUG(errs() << "Enter: " << M.getModuleIdentifier() << "\n\n");

    // Do not change the code
    return false;
  }


  bool doFinalization(Module &M) override {
    DEBUG(errs() << "Exit: " << M.getModuleIdentifier() << "\n\n");

    // Do not change the code
    return false;
  }


  /// Visit all the module content
  bool runOnModule(Module &M) override {
    for (auto &F : M.functions()) {
      DEBUG(errs() << "Function: ";
            errs().write_escaped(F.getName()) << '\n');
      // Detect functions that did not specify with noinline attribute called in SYCL
      // kernel
      for (Use &U : F.uses()) {
        CallSite CS(U.getUser());
        if (CS.getInstruction() != nullptr) {
          if (sycl::isKernel(*(CS.getInstruction()->getParent()->getParent()))) {
            DEBUG(dbgs() << F.getName() << " is a function called in kernel.\n");
            if (F.hasLocalLinkage() && !F.hasFnAttribute(Attribute::NoInline)) {
              // Add always inline attribute. This can force the function to be 
        // inline in inline pass
        F.addFnAttr(Attribute::AlwaysInline);
        DEBUG(dbgs() << F.getName()<< " add AlwaysInline attribute.\n");
      }
    }
        }
      }
    }


    // The module probably changed
    return true;
  }
};


}

char SYCLFunctionInKernelAlwaysInline::ID = 0;
INITIALIZE_PASS_BEGIN(SYCLFunctionInKernelAlwaysInline, "SYCL-function-in-kernel-always-inline",
                                          "SYCL function in kernel always inline pass", false, false)
INITIALIZE_PASS_END(SYCLFunctionInKernelAlwaysInline, "SYCL-function-in-kernel-always-inline",
                                          "SYCL function in kernel always inline pass", false, false)

Pass *llvm::createSYCLFunctionInKernelAlwaysInlinePass() { return new SYCLFunctionInKernelAlwaysInline(); }
