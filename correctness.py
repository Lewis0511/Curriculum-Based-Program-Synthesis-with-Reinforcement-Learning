import multiprocessing
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
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, state): return self.network(torch.FloatTensor(state))

class IRProgramEnvironment:


    def __init__(self, max_blocks=8, max_instructions=32, max_bool_registers=4, max_int_registers=4, max_ptr_registers=4):
        self.module = None
        self.main_function = None
        self.builder = None
        self.rewards_queue = None

        self.action = None
        self.blocks = None
        self.bool_registers = None
        self.int_registers = None
        self.ptr_registers = None
        self.instructions = None

        self.max_blocks = max_blocks
        self.max_instructions = max_instructions
        self.max_bool_registers = max_bool_registers
        self.max_int_registers = max_int_registers
        self.max_ptr_registers = max_ptr_registers
        self.strides = [768, 256, 64, 16, 4, 1]
        self.instructions_space = ["icmp slt", "icmp sgt", "branch", "cbranch", "store", "load", "position_at_end", "getelementptr"]

    def reset(self):
        self.module = ir.Module(name="module")
        function_type = ir.FunctionType(ir.VoidType(), [ir.PointerType(ir.IntType(32))])
        self.main_function = ir.Function(self.module, function_type, name="main")
        self.main_function.args[0].name = "arr"
        block = self.main_function.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        self.rewards_queue = multiprocessing.Queue()

        self.action = -1
        self.blocks = [block]
        self.bool_registers = []
        self.int_registers = []
        self.ptr_registers = []
        self.instructions = []
        return self.get_state()

    def get_state(self):
        # state = [block_states_mask, num_bool_registers, num_int_registers, num_ptr_registers]
        # state_size == 8 * 4 * 4 * 4 == 512
        return np.array([len(self.blocks), len(self.bool_registers), len(self.int_registers), len(self.ptr_registers)])

    def execute(self):
        try:
            self.builder.ret_void()  # TODO: this is one of the major problems
            with open("output.ll", "w") as f: f.write(str(self.module))
            target = llvm.Target.from_default_triple()
            target_machine = target.create_target_machine()
            compiled_module = llvm.parse_assembly(str(self.module))
            execution_engine = llvm.create_mcjit_compiler(compiled_module, target_machine)

            main_pointer = execution_engine.get_function_address("main")
            main_function = CFUNCTYPE(None, POINTER(c_int))(main_pointer)
            permutations = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
            reward = 0  # 1.0 * len(self.instructions) / self.max_instructions

            for permutation in permutations:
                arr = (c_int * 3)(*permutation)
                sorted_arr = sorted(permutation)
                main_function(arr)
                reward += (arr[0] == sorted_arr[0] and arr[1] == sorted_arr[1] and arr[2] == sorted_arr[2]) / 6.0

            self.rewards_queue.put(reward)

        except Exception as e:
            print(f"execute: {str(self.module)}\n\n{e}")
            self.rewards_queue.put(0)

    def validate(self):
        process = multiprocessing.Process(target=self.execute)
        process.start()
        process.join(timeout=5)
        if not process.is_alive(): return self.rewards_queue.get()
        process.terminate()
        process.join()
        return 0

# The code below is the expected LLVM IR program to be generated

# arr_0_ptr = self.builder.gep(self.registers[0], [ir.Constant(ir.IntType(32), 0)], name="arr_0_ptr")
# arr_1_ptr = self.builder.gep(self.registers[0], [ir.Constant(ir.IntType(32), 1)], name="arr_1_ptr")
# arr_2_ptr = self.builder.gep(self.registers[0], [ir.Constant(ir.IntType(32), 2)], name="arr_2_ptr")
# arr_0 = self.builder.load(arr_0_ptr, name="arr_0")
# arr_1 = self.builder.load(arr_1_ptr, name="arr_1")
# arr_2 = self.builder.load(arr_2_ptr, name="arr_2")

# comp_01 = self.builder.icmp_signed(">", arr_0, arr_1, name="comp_01")
# swap_01 = self.main_function.append_basic_block("swap_01")
# cont_01 = self.main_function.append_basic_block("cont_01")
# self.builder.cbranch(comp_01, swap_01, cont_01)
# self.builder.position_at_end(swap_01)
# self.builder.store(arr_0, arr_1_ptr)
# self.builder.store(arr_1, arr_0_ptr)
# self.builder.branch(cont_01)
# self.builder.position_at_end(cont_01)

# comp_02 = self.builder.icmp_signed(">", arr_0, arr_2, name="comp_02")
# swap_02 = self.main_function.append_basic_block("swap_02")
# cont_02 = self.main_function.append_basic_block("cont_02")
# self.builder.cbranch(comp_02, swap_02, cont_02)
# self.builder.position_at_end(swap_02)
# self.builder.store(arr_0, arr_2_ptr)
# self.builder.store(arr_2, arr_0_ptr)
# self.builder.branch(cont_02)
# self.builder.position_at_end(cont_02)

# comp_12 = self.builder.icmp_signed(">", arr_1, arr_2, name="comp_12")
# swap_12 = self.main_function.append_basic_block("swap_12")
# cont_12 = self.main_function.append_basic_block("cont_12")
# self.builder.cbranch(comp_12, swap_12, cont_12)
# self.builder.position_at_end(swap_12)
# self.builder.store(arr_1, arr_2_ptr)
# self.builder.store(arr_2, arr_1_ptr)
# self.builder.branch(cont_12)
# self.builder.position_at_end(cont_12)
# self.builder.ret_void()

    def step(self, action):
        # action = [instruction_index, array_index, register_index_0, register_index_1, block_index_0, block_index_1]
        # action_size == 8 * 3 * 4 ** 2 * 4 ** 2 == 6,144

        self.action = action
        instruction_index, action = action // self.strides[0], action % self.strides[0]
        array_index, action = action // self.strides[1], action % self.strides[1]
        register_index_0, action = action // self.strides[2], action % self.strides[2]
        register_index_1, action = action // self.strides[3], action % self.strides[3]
        block_index_0, action = action // self.strides[4], action % self.strides[4]
        block_index_1, action = action // self.strides[5], action % self.strides[5]
        self.instructions.append(instruction_index)
        instruction = self.instructions_space[instruction_index]
        register_indices = [register_index_0, register_index_1]
        block_indices = [block_index_0, block_index_1]

        try:
            if instruction in ["icmp slt", "icmp sgt"]:
                new_bool_register = f"bool_{len(self.bool_registers)}"
                sign = "<" if instruction == "icmp slt" else ">"
                val0 = self.int_registers[register_index_0]
                val1 = self.int_registers[register_index_1]
                result = self.builder.icmp_signed(sign, val0, val1, name=new_bool_register)
                self.bool_registers.append(result)

            elif instruction == "branch":
                block = self.blocks[block_index_0]
                self.builder.branch(block)

            elif instruction == "cbranch":
                condition = self.bool_registers[register_index_0]
                true_block = self.main_function.append_basic_block(f"true_{len(self.blocks) // 2}")
                false_block = self.main_function.append_basic_block(f"false_{len(self.blocks) // 2}")
                self.builder.cbranch(condition, true_block, false_block)
                self.builder.position_at_end(true_block)
                self.blocks.append(true_block)
                self.blocks.append(false_block)

            elif instruction == "store":
                val = self.int_registers[register_index_0]
                ptr = self.ptr_registers[register_index_1]
                self.builder.store(val, ptr)

            elif instruction == "load":
                new_int_register = f"int_{len(self.int_registers)}"
                ptr = self.ptr_registers[register_index_0]
                val = self.builder.load(ptr, name=new_int_register)
                self.int_registers.append(val)

            elif instruction == "position_at_end":
                block = self.blocks[block_index_0]
                self.builder.position_at_end(block)

            elif instruction == "getelementptr":
                # we always get a pointer to an element in the array of the main function's argument fot this task
                new_ptr_register = f"ptr_{len(self.ptr_registers)}"
                ptr = self.builder.gep(self.main_function.args[0], [ir.Constant(ir.IntType(32), array_index)], name=new_ptr_register)
                self.ptr_registers.append(ptr)

            reward = self.validate()
            finished = len(self.blocks) >= self.max_blocks
            if len(self.instructions) >= self.max_instructions: finished = True
            elif len(self.bool_registers) >= self.max_bool_registers: finished = True
            elif len(self.int_registers) >= self.max_int_registers: finished = True
            elif len(self.ptr_registers) >= self.max_ptr_registers: finished = True
            print(f"action: {[instruction_index, array_index, register_indices, block_indices]}, reward: {reward}, finished: {finished}")
            return self.get_state(), reward, finished

        except Exception as e:
            print(f"action: {[instruction_index, array_index, register_indices, block_indices]}")
            print(f"step: {str(self.module)}\n\n{instruction}: {e}")
            return self.get_state(), 0, True

def DQN_train(num_episodes=1000, max_blocks=8, max_instructions=32, max_bool_registers=4, max_int_registers=4, max_ptr_registers=4):
    rewards = []
    success_rates = []
    num_correct_llvm_ir_code = 0

    epsilon_start = 0.99
    epsilon_end = 0.90
    discount_factor = 0.99
    input_dim = 4
    output_dim = 8 * 3 * 4 ** 2 * 4 ** 2
    # action = [instruction_index, array_index, register_index_0, register_index_1, block_index_0, block_index_1]
    # action_size == 8 * 3 * 4 ** 2 * 4 ** 2 == 6,144

    agent = DQN(input_dim=input_dim, output_dim=output_dim)
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-3)
    env = IRProgramEnvironment(max_blocks, max_instructions, max_bool_registers, max_int_registers, max_ptr_registers)

    for episode in range(num_episodes):
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * episode / num_episodes  # TODO: linear exploration rate for now
        curr_state = env.reset()
        finished = False

        while not finished:
            action = None
            print(f"Episode {episode}, number of instructions: {len(env.instructions)}")
            if random.random() < epsilon:
                action_space = []
                block_indices = [index for index, block in enumerate(env.blocks[1:]) if not block.is_terminated]

                if len(env.int_registers) >= 2:
                    instruction_index = random.randrange(2)
                    register_index_0, register_index_1 = random.sample(range(len(env.int_registers)), 2)
                    action = instruction_index * env.strides[0] + register_index_0 * env.strides[2] + register_index_1 * env.strides[3]
                    action_space.append(action)

                if block_indices:
                    instruction_index = 2
                    block_index_0 = random.choice(block_indices)
                    action = instruction_index * env.strides[0] + block_index_0 * env.strides[4]
                    action_space.append(action)

                if len(env.bool_registers) >= 1:
                    instruction_index = 3
                    register_index_0 = random.randrange(len(env.bool_registers))
                    action = instruction_index * env.strides[0] + register_index_0 * env.strides[2]
                    action_space.append(action)

                if len(env.int_registers) >= 1 and len(env.ptr_registers) >= 1:
                    instruction_index = 4
                    register_index_0 = random.randrange(len(env.int_registers))
                    register_index_1 = random.randrange(len(env.ptr_registers))
                    action = instruction_index * env.strides[0] + register_index_0 * env.strides[2] + register_index_1 * env.strides[3]
                    action_space.append(action)

                if len(env.ptr_registers) >= 1:
                    instruction_index = 5
                    register_index_0 = random.randrange(len(env.ptr_registers))
                    action = instruction_index * env.strides[0] + register_index_0 * env.strides[2]
                    action_space.append(action)

                if block_indices:
                    instruction_index = 6
                    block_index_0 = random.choice(block_indices)
                    action = instruction_index * env.strides[0] + block_index_0 * env.strides[4]
                    action_space.append(action)

                if True:
                    instruction_index = 7
                    array_index = random.randrange(3)
                    action = instruction_index * env.strides[0] + array_index * env.strides[1]
                    action_space.append(action)

                # index    instructions     block    bool    int    ptr
                #  0/1     icmp slt/sgt                       2
                #   2      branch             1
                #   3      cbranch                    1
                #   4      store                              1      1
                #   5      load                                      1
                #   6      position_at_end    1
                #   7      getelementptr
                action = random.choice(action_space)

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
                predicted_Q_value = agent(curr_state)[action]
                loss = (predicted_Q_value - target_Q_value) ** 2
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            rewards.append(reward)
            success_rates.append(reward)
            if reward >= 1.0 / 6.0:  # TODO: unable to generate fully correct IIVM IR program for now
                num_correct_llvm_ir_code += 1
                with open("correctness.txt", "a") as f: f.write(f"Episode: {episode}, reward: {reward}, correct generated LLVM IR code: \n{str(env.module)}\n")

        if episode % 100 == 0: print(f"Finished episode {episode} of {num_episodes}, number of correct LLVM IR code: {num_correct_llvm_ir_code}")

    torch.save(agent.state_dict(), "correctness.pth")
    with open("correctness.txt", "a") as f: f.write(f"Finished: Number of correct generated LLVM IR code = {num_correct_llvm_ir_code}\n\n")
    return rewards, success_rates

if __name__ == "__main__":
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