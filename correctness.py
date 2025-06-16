import os
import random

import llvmlite.binding as llvm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ctypes import c_int, CFUNCTYPE, POINTER
from llvmlite import ir

random.seed()
llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

class DQN(nn.Module):


    def __init__(self, input_dim, output_dim):

        super(DQN, self).__init__()
        self.net = nn.Sequential(  # TODO
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

    def forward(self, state): return self.net(torch.FloatTensor(state))


class IRProgramEnvironment:


    def __init__(self, max_blocks=4, max_registers=4, max_instructions=16):

        self.module = None
        self.main_function = None
        self.builder = None

        self.blocks = None
        self.registers = None
        self.instructions = None
        self.action = None

        self.max_blocks = max_blocks
        self.max_registers = max_registers
        self.max_instructions = max_instructions
        self.strides = [3072, 768, 192, 48, 12, 3, 1]  # TODO
        self.instructions_space = ["icmp slt", "icmp sgt", "br", "select", "store", "load", "phi", "getelementptr", "ret"]

    def get_state(self):

        action = self.action
        state = [None] * len(self.strides)
        for i in range(7): state[i], action = action // self.strides[i], action % self.strides[i]
        return torch.from_numpy(np.array(state).astype(np.float32))

    def reset(self):

        self.module = ir.Module(name="module")
        function_type = ir.FunctionType(ir.VoidType(), [ir.PointerType(ir.IntType(32))])
        self.main_function = ir.Function(self.module, function_type, name="main")
        self.main_function.args[0].name = "arr"
        block = self.main_function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)

        self.blocks = [block]
        self.registers = [self.main_function.args[0]]
        self.instructions = []
        self.action = -1

        return self.get_state()

    def validate_program(self):

        try:

            with open("output.ll", "w") as f: f.write(str(self.module))
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            compiled_module = llvm.parse_assembly(str(self.module))
            execution_engine = llvm.create_mcjit_compiler(compiled_module, target_machine)

            main_pointer = execution_engine.get_function_address("main")
            main_function = CFUNCTYPE(None, POINTER(c_int))(main_pointer)
            permutations = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
            correct_count = 0

            for permutation in permutations:
                arr = (c_int * 3)(*permutation)
                main_function(arr)
                correct_count += arr[0] <= arr[1] <= arr[2]
            reward = correct_count / 6.0
            # reward = 0
            # reward += 0 in self.instructions
            # reward += 1 in self.instructions
            # reward += 2 in self.instructions
            # reward += 3 in self.instructions
            # reward += 4 in self.instructions
            # reward += 5 in self.instructions
            # reward += 6 in self.instructions
            # reward += 7 in self.instructions
            return reward

        except Exception as e:
            # print(str(self.module))
            # print("validate:", e)
            return 0

    def step(self, action):

        self.action = action
        instruction_index, action = action // self.strides[0], action % self.strides[0]
        array_index, action = action // self.strides[1], action % self.strides[1]
        registers_index_0, action = action // self.strides[2], action % self.strides[2]
        registers_index_1, action = action // self.strides[3], action % self.strides[3]
        registers_index_2, action = action // self.strides[4], action % self.strides[4]
        blocks_index_0, action = action // self.strides[5], action % self.strides[5]
        blocks_index_1, action = action // self.strides[6], action % self.strides[6]

        instruction = self.instructions_space[instruction_index]
        registers_indices = [registers_index_0, registers_index_1, registers_index_2]
        blocks_indices = [blocks_index_0, blocks_index_1]
        self.instructions.append(instruction_index)
        new_register = f"{len(self.registers) - 1}"

        try:

            if instruction in ["icmp slt", "icmp sgt"]:

                sign = "<" if instruction == "icmp slt" else ">"
                val0 = self.registers[registers_indices[0]]
                val1 = self.registers[registers_indices[1]]
                result = self.builder.icmp_signed(sign, val0, val1, name=new_register)
                self.registers.append(result)

            elif instruction == "br":

                condition = self.registers[registers_indices[0]]
                false_block = self.main_function.append_basic_block(f"false_{len(self.blocks)}")
                true_block = self.main_function.append_basic_block(f"true_{len(self.blocks)}")
                merge_block = self.main_function.append_basic_block(f"merge_{len(self.blocks)}")
                self.builder.cbranch(condition, false_block, true_block)

                self.builder.position_at_end(false_block)
                self.builder.branch(merge_block)
                self.builder.position_at_end(true_block)
                self.builder.branch(merge_block)
                self.builder.position_at_end(merge_block)

                self.blocks.append(true_block)
                self.blocks.append(false_block)
                self.blocks.append(merge_block)

            elif instruction == "select":

                condition = self.registers[registers_indices[2]]
                val0 = self.registers[registers_indices[0]]
                val1 = self.registers[registers_indices[1]]
                result = self.builder.select(condition, val0, val1, name=new_register)
                self.registers.append(result)

            elif instruction == "store":

                # we always store the value to the array of the main function's argument for this task
                val = self.registers[registers_indices[0]]
                ptr = self.registers[0]
                self.builder.store(val, ptr)

            elif instruction == "load":

                # we always load the value of the array of the main function's argument for this task
                ptr = self.registers[0]
                result = self.builder.load(ptr, name=new_register)
                self.registers.append(result)

            elif instruction == "phi":

                val0 = self.registers[registers_indices[0]]
                val1 = self.registers[registers_indices[1]]
                block0 = self.blocks[blocks_indices[0]]
                block1 = self.blocks[blocks_indices[1]]

                result = self.builder.phi(ir.IntType(32), name=new_register)
                result.add_incoming(val0, block0)
                result.add_incoming(val1, block1)
                self.registers.append(result)

            elif instruction == "getelementptr":

                # we always get an element from the array of the main function's argument for this task
                result = self.builder.gep(self.registers[0], [ir.Constant(ir.IntType(32), array_index)], name=new_register)
                self.registers.append(result)

            elif instruction == "ret":

                self.builder.ret_void()

            reward = self.validate_program()
            finished = len(self.blocks) >= self.max_blocks or len(self.registers) >= self.max_registers or len(self.instructions) >= self.max_instructions
            return self.get_state(), reward, finished or instruction == "ret"

        except Exception as e:
            # print(str(self.module))
            # print("step:", e)
            return self.get_state(), 0, True

def DQN_train(num_episodes=1000, max_blocks=4, max_registers=4, max_instructions=16):

    rewards = []
    success_rates = []
    num_correct_llvm_ir_code = 0

    epsilon_start = 0.99
    epsilon_end = 0.90
    discount_factor = 0.99
    output_dim = 9 * 3 * 4 ** 3 * 4 ** 2

    agent = DQN(input_dim=7, output_dim=output_dim)
    env = IRProgramEnvironment(max_blocks=max_blocks, max_registers=max_registers, max_instructions=max_instructions)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)

    for episode in range(num_episodes):

        epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / num_episodes  # TODO: linear exploration rate for now
        curr_state = env.reset()
        reward = 0
        finished = False

        while not finished:

            if random.random() < epsilon: action = random.randrange(0, output_dim)
            else:
                with torch.no_grad():
                    curr_Q_value = agent(curr_state)
                    action = curr_Q_value.argmax().item()
            next_state, reward, finished = env.step(action)
            curr_state = next_state

            if not finished:
                with torch.no_grad():
                    next_Q_value = agent(next_state)
                target_Q_value = reward + discount_factor * next_Q_value.max().item()
                predicted_Q_value = agent(torch.Tensor(curr_state))[action]
                loss = (predicted_Q_value - target_Q_value) ** 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            rewards.append(reward)
            success_rates.append(reward)
            if reward >= 1.0 / 6.0:  # TODO
                num_correct_llvm_ir_code += 1
                with open("correctness.txt", "a") as f: f.write(f"Episode: {episode}, reward: {reward}, correct generated LLVM IR code: \n{str(env.module)}\n")

        if episode % 100 == 0: print(f"Finished episode {episode} of {num_episodes}")

    torch.save(agent.state_dict(), "correctness.pth")
    with open("correctness.txt", "a") as f: f.write(f"Finished: Number of correct generated LLVM IR code = {num_correct_llvm_ir_code}\n\n")
    return rewards, success_rates

if os.path.exists("correctness.txt"): os.remove("correctness.txt")
rewards, success_rates = DQN_train()

with open("correctness.txt", "a") as f:
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
plt.savefig("correctness.png")

# def moving_average(data, window=50):
#     return np.convolve(data, np.ones(window) / window, mode='valid')

# episodes = np.arange(len(rewards))
# plt.subplot(1, 2, 1)
# plt.plot(episodes, rewards, alpha=0.3, label="Raw Rewards")
# plt.plot(episodes[:len(moving_average(rewards))], moving_average(rewards), label="Smoothed Rewards")
# plt.legend()

# plt.subplot(1, 2, 2)
# success_rate = np.cumsum(success_rates) / (np.arange(len(success_rates)) + 1)
# plt.plot(episodes, success_rate, alpha=0.3, label="Raw Success Rate")
# plt.plot(episodes[:len(moving_average(success_rate))], moving_average(success_rate), label="Smoothed Success Rate")
# plt.legend()

# Episode: 156, reward: 1.0, valid generated LLVM IR code:
# ; ModuleID = "Module"
# target triple = "unknown-unknown-unknown"
# target datalayout = ""

# define void @"main"(i32* %".1")
# {
# entry:
#   %"0" = load i32, i32* %".1"
#   store i32 0, i32* %".1"
#   %"1" = select  i1 false, i32 0, i32 %"0"
#   %"2" = icmp sgt i32 %"0", %"1"
#   %"3" = load i32, i32* %".1"
#   %"4" = getelementptr i32, i32* %".1", i32 1
#   store i32 %"3", i32* %"4"
#   %"5" = icmp slt i32 %"3", 0
#   %"6" = icmp sgt i32 0, 0
#   %"7" = getelementptr i32, i32* %".1", i32 1
#   %"8" = icmp slt i32 0, 0
#   ret void
# }

# Episode: 314, reward: 1.0, valid generated LLVM IR code:
# ; ModuleID = "Module"
# target triple = "unknown-unknown-unknown"
# target datalayout = ""

# define void @"main"(i32* %".1")
# {
# entry:
#   %"0" = getelementptr i32, i32* %".1", i32 1
#   store i32 0, i32* %"0"
#   %"1" = load i32, i32* %"0"
#   store i32 0, i32* %".1"
#   %"2" = icmp slt i32 0, %"1"
#   ret void
# }

# Episode: 866, reward: 1.0, valid generated LLVM IR code:
# ; ModuleID = "Module"
# target triple = "unknown-unknown-unknown"
# target datalayout = ""

# define void @"main"(i32* %".1")
# {
# entry:
#   store i32 0, i32* %".1"
#   %"0" = getelementptr i32, i32* %".1", i32 1
#   br i1 false, label %"true_3", label %"false_3"
# false_3:
#   br label %"merge_3"
# true_3:
#   br label %"merge_3"
# merge_3:
#   br i1 false, label %"true_4", label %"false_4"
# false_4:
#   br label %"merge_4"
# true_4:
#   br label %"merge_4"
# merge_4:
#   store i32 0, i32* %"0"
#   br i1 false, label %"true_6", label %"false_6"
# false_6:
#   br label %"merge_6"
# true_6:
#   br label %"merge_6"
# merge_6:
#   br i1 false, label %"true_7", label %"false_7"
# false_7:
#   br label %"merge_7"
# true_7:
#   br label %"merge_7"
# merge_7:
#   store i32 0, i32* %"0"
#   %"1" = load i32, i32* %"0"
#   store i32 0, i32* %".1"
#   %"2" = getelementptr i32, i32* %".1", i32 0
#   %"3" = select  i1 false, i32 %"1", i32 0
#   %"4" = select  i1 false, i32 0, i32 %"3"
#   br i1 false, label %"true_14", label %"false_14"
# false_14:
#   br label %"merge_14"
# true_14:
#   br label %"merge_14"
# merge_14:
#   ret void
# }