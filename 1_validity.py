import llvmlite.binding as llvm
import numpy as np
import random
import torch
import torch.nn as nn

from ctypes import CFUNCTYPE
from llvmlite import ir

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state): return self.net(torch.FloatTensor(state))

class IRProgramEnvironment:

    def __init__(self, max_instructions=10):
        self.module = None
        self.builder = None
        self.operands = None
        self.instruction_count = None
        self.max_instructions = max_instructions
        self.action_space = ["add", "sub", "mul", "icmp sgt", "ret"]

    def get_state(self): return np.array([self.instruction_count, len(self.operands)])

    def reset(self):
        self.module = ir.Module(name="module")
        function_type = ir.FunctionType(ir.VoidType(), ())
        main_function = ir.Function(self.module, function_type, name="main")
        block = main_function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self.operands = []
        self.instruction_count = 0
        return self.get_state()

    def validate_program(self):
        try:
            with open("output.ll", "w") as f: f.write(str(self.module))
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            compiled_module = llvm.parse_assembly(str(self.module))
            execution_engine = llvm.create_mcjit_compiler(compiled_module, target_machine)
            main_ptr = execution_engine.get_function_address("main")
            main_func = CFUNCTYPE(None)(main_ptr)
            main_func()
            return 1
        except Exception:  # TODO: Too broad exception clause. Do not know how to silent this warning appropriately.
            return 0

    def step(self, action_index):
        self.instruction_count += 1
        action = self.action_space[action_index]
        finished = self.instruction_count >= self.max_instructions or action == "ret"
        i32 = ir.IntType(32)
        new_operand = f"{len(self.operands)}"
        if action in ["add", "sub", "mul"]:
            op1 = self.operands[-2] if len(self.operands) >= 2 else ir.Constant(i32, 1)
            op2 = self.operands[-1] if len(self.operands) >= 1 else ir.Constant(i32, 2)
            if action == "add": result = self.builder.add(op1, op2, new_operand)
            if action == "sub": result = self.builder.sub(op1, op2, new_operand)
            if action == "mul": result = self.builder.mul(op1, op2, new_operand)
            self.operands.append(result)
        elif action == "icmp sgt":
            op1 = self.operands[-2] if len(self.operands) >= 2 else ir.Constant(i32, 1)
            op2 = self.operands[-1] if len(self.operands) >= 1 else ir.Constant(i32, 2)
            result = self.builder.icmp_signed(">", op1, op2, new_operand)
            result = self.builder.zext(result, i32, new_operand)
            self.operands.append(result)
        elif action == "ret":
            self.builder.ret_void()
        reward = self.validate_program()
        return self.get_state(), reward, finished

env = IRProgramEnvironment(max_instructions=10)
agent = DQN(input_dim=2, output_dim=len(env.action_space))
num_valid_llvm_ir_code = 0
epsilon = 0.01

for episode in range(1000):
    finished = False
    curr_state = env.reset()
    while not finished:
        if random.random() < epsilon:
            action = random.randint(0, len(env.action_space) - 1)
        else:
            with torch.no_grad():
                Q_value = agent(curr_state)
                action = Q_value.argmax().item()
        next_state, reward, finished = env.step(action)  # TODO: Name 'action' can be undefined. Do not know how to silent this warning.
        curr_state = next_state
        if reward == 1:
            print(f"Episode: {episode}, reward: {reward}, valid generated LLVM IR code: \n{str(env.module)}")
            num_valid_llvm_ir_code += 1
            break

print(f"Finished: Number of valid generated LLVM IR code = {num_valid_llvm_ir_code}")