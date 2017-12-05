//===- SYCLModifySPIRFuncName.cpp                               ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Detect functions called in SYCL kernels and modify their names.  
// ===----------------------------------------------------------------------===//

#include "llvm/ADT/Statistic.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/CallSite.h"
#include "llvm/Pass.h"
#include "llvm/SYCL.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/CtorUtils.h"

using namespace llvm;

/// Switch on debug with set DebugFlag=0 or set DebugFlag=1 in debugger or with
/// option -debug or -debug-only=SYCL-modify-SPIR-funcName
#define DEBUG_TYPE "SYCL-modify-SPIR-func-name"


/// Displayed with -stats
STATISTIC(SYCLFuncCalledInKernelFound, "Number of SYCL kernel functions");

// Put the code in an anonymous namespace to avoid polluting the global
// namespace
namespace {

/// Detect functions called in SYCL kernels and modify their names.
struct SYCLModifySPIRFuncName : public ModulePass {

  static char ID; // Pass identification, replacement for typeid


  SYCLModifySPIRFuncName() : ModulePass(ID) {}


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
    // count is for naming new name for each function called in kernel 
    int count = 0;

    for (auto &F : M.functions()) {
      // Only consider definition of functions
      if (!F.isDeclaration()) {
        // If functions are called by kernel, modify their names
        for (Use &U : F.uses()) {
          CallSite CS(U.getUser());
          if (CS.getInstruction() != nullptr) {
            if (sycl::isKernel(*(CS.getInstruction()->getParent()->getParent()))) {
              DEBUG(dbgs() << F.getName() << "is called in kernel function. Force to change name.\n");
              SYCLFuncCalledInKernelFound++;
              F.setName("foo." + Twine(count++));  
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

char SYCLModifySPIRFuncName::ID = 0;
static RegisterPass<SYCLModifySPIRFuncName> X { "SYCL-modify-SPIR-func-name",
                                          "SYCL modify SPIR function name pass" };
