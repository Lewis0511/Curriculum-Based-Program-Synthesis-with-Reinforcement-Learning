import os
import random

import llvmlite.binding as llvm
import matplotlib.pyplot as plt
import numpy as np
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
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, state): return self.net(torch.FloatTensor(state))

class IRProgramEnvironment:


    def __init__(self, max_instructions=10):
        self.module = None
        self.builder = None
        self.operands = None
        self.instruction_count = None
        self.last_action_index = None
        self.max_instructions = max_instructions
        self.action_space = ["add", "sub", "mul", "icmp sgt", "ret"]

    def get_state(self):
        opcode_one_hot = [0] * len(self.action_space)
        if self.last_action_index is not None: opcode_one_hot[self.last_action_index] = 1
        # return np.array([self.instruction_count, len(self.operands)])  This is the initial simpler state representation
        return np.array([1.0 * self.instruction_count / self.max_instructions, 1.0 * len(self.operands) / self.max_instructions] + opcode_one_hot)

    def reset(self):
        self.module = ir.Module(name="module")
        function_type = ir.FunctionType(ir.VoidType(), ())
        main_function = ir.Function(self.module, function_type, name="main")
        block = main_function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self.operands = []
        self.instruction_count = 0
        self.last_action_index = None
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
        self.last_action_index = action_index
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

def DQN_train(num_episodes=1000, max_instructions=10):
    rewards = []
    success_rates = []
    num_valid_llvm_ir_code = 0
    env = IRProgramEnvironment(max_instructions=max_instructions)
    epsilon = 0.99  # TODO: 0.99 seems to work best from an exploration perspective
    agent = DQN(input_dim=2 + len(env.action_space), output_dim=len(env.action_space))
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    for episode in range(num_episodes):
        finished = False
        total_reward = 0
        curr_state = env.reset()
        while not finished:
            if random.random() < epsilon:
                action = random.randrange(0, len(env.action_space))
            else:
                with torch.no_grad():
                    curr_Q_value = agent(curr_state)
                    action = curr_Q_value.argmax().item()
            next_state, reward, finished = env.step(action)
            curr_state = next_state
            total_reward += reward
            if not finished:
                with torch.no_grad():
                    next_Q_value = agent(next_state).max().item()
                target_Q_value = reward + 0.99 * next_Q_value  # 0.99 is the discount factor
                predicted_Q_value = agent(torch.FloatTensor(curr_state))[action]
                loss = (predicted_Q_value - target_Q_value) ** 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        rewards.append(total_reward)
        success_rates.append(1 if total_reward > 0 else 0)
        if total_reward > 0:
            with open("1_validity.txt", "a") as f:
                f.write(f"Episode: {episode}, total_reward: {total_reward}, valid generated LLVM IR code: \n{str(env.module)}\n")
            num_valid_llvm_ir_code += 1

    torch.save(agent.state_dict(), "1_validity.pt")

    with open("1_validity.txt", "a") as f:
        f.write(f"Finished: Number of valid generated LLVM IR code = {num_valid_llvm_ir_code}\n\n")

    return rewards, success_rates

os.remove("1_validity.txt")
rewards, success_rates = DQN_train()

with open("1_validity.txt", "a") as f:
    f.write(f"Rewards: {rewards}\n\n")
    f.write(f"Success rates: {success_rates}")

plt.figure(figsize=(16, 8))

plt.subplot(1, 2, 1)
plt.plot(rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Rewards Curve")

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(success_rates) / (np.arange(len(success_rates)) + 1), label="Success Rate")
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title("Success Rate Curve")

plt.tight_layout()
plt.savefig("1_validity.png")