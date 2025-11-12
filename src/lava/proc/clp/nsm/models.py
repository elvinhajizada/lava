# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: BSD-3-Clause
# See: https://spdx.org/licenses/

import numpy as np
import random
random.seed(42)  # Set your desired seed here

from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol
from lava.magma.core.model.py.ports import PyInPort, PyOutPort
from lava.magma.core.model.py.type import LavaPyType
from lava.magma.core.resources import CPU
from lava.magma.core.decorator import implements, requires, tag
from lava.magma.core.model.py.model import PyLoihiProcessModel

from lava.proc.clp.nsm.process import Readout
from lava.proc.clp.nsm.process import Allocator


@implements(proc=Readout, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyReadoutModel(PyLoihiProcessModel):
    """Python implementation of the Readout process.
    This process will run in super host and will be the main interface
    process with the user.
    """
    inference_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    label_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)

    user_output: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    trigger_alloc: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32, precision=24)
    feedback: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    proto_labels: np.ndarray = LavaPyType(np.ndarray, np.int32)
    last_winner_id: np.int32 = LavaPyType(np.ndarray, np.int32)
    testing: np.int32 = LavaPyType(np.ndarray, np.int32)
    supervised: np.int32 = LavaPyType(np.ndarray, np.int32)
    n_steps_per_sample: np.int32 = LavaPyType(np.ndarray, np.int32)
    verbose: np.int32 = LavaPyType(np.ndarray, np.int32)

    def run_spk(self) -> None:
        if (self.time_step % self.n_steps_per_sample == 1) and self.verbose >= 1:
            print("-----------------------------------------------------------------")
            print(self.time_step // self.n_steps_per_sample)
        if self.verbose == 3:
            print("---------------------")
            print("Time step:", self.time_step)

        # Read the user-provided label
        # print("trying to recv label")
        user_label = self.label_in.recv()[0]
        # print("Time", ((self.time_step-1) % 25) + 1 , "User labels are read:", user_label)
        # print("User labels are read:", user_label)

        # Read the output of the prototype neurons
        # print("trying to proto out")
        output_vec = self.inference_in.recv()
        # if output_vec.sum() > 0:
        #     print("Prototype outputs are read:", output_vec)
        # if self.inference_in.probe():
        #     output_vec = self.inference_in.recv()
        #     # print("Prototype outputs are read:", output_vec)
        # else: 
        #     # print("No proto output")
        #     output_vec = np.zeros(shape=self.inference_in.shape)
        
        # Feedback about the correctness of prediction. +1 if correct,
        # -1 if incorrect, 0 if no label is provided by the user at this point.
        infer_check = 0

        # If there is an active prototype neuron, this will temporarily store
        # the label of that neuron
        inferred_label = 0

        # Flag for allocation trigger
        allocation_trigger = False

        next_alloc_id = (self.proto_labels == 0).argmax() if 0 in self.proto_labels else -1

        # If any prototype neuron is active, then we go here. We assume there
        # is only one neuron active in the prototype population

        # Define the overflown values to check and remove
        overflown_values = [1,256,257, 65536, 65537, 65792, 65793, 16777216, 16777217, 16777471,16777472]

        # Use np.isin to create a mask that is True for values NOT in overflown_values
        mask = ~np.isin(output_vec, overflown_values)

        # Filter the array using the mask
        output_vec = output_vec[mask] 

        
        
        if output_vec.any():
            if self.verbose >= 2:
                print("time step:", int((self.time_step-1) % self.n_steps_per_sample)+1)
                print("output_vec:", output_vec)
            curr_output = output_vec[np.nonzero(output_vec)] - 2
            # print(curr_output)
            # Get labels for each ID in `ids`
            voted_labels = self.proto_labels[curr_output]
            # print(voted_labels)

            if 0 not in voted_labels:
                if len(voted_labels) == 1:
                    # Only one label voted, take it directly
                    inferred_label = voted_labels[0]
                    self.last_winner_id = curr_output[0]
                else:
                    # Count occurrences of each label
                    # print("Check 1")
                    label_counts = np.bincount(voted_labels)
                    max_count = label_counts.max()
                    # print("Check 2")
                    # Find all labels with the maximum count
                    candidates = np.flatnonzero(label_counts == max_count)

                    # Randomly select one of the labels with the maximum count
                    most_common_label = random.choice(candidates)
                    # print("Check 3")
                    # Find IDs corresponding to the most common label
                    prototype_ids_with_winner_label = np.where(self.proto_labels[0:next_alloc_id+1] == most_common_label)[0]

                    # Randomly select one ID among the prototypes with the winning label
                    chosen_prototype_id = random.choice(prototype_ids_with_winner_label)
                    
                    self.last_winner_id = chosen_prototype_id
                    # print("Check 4")
                    # Get the label of this neuron from the labels' list
                    inferred_label = self.proto_labels[self.last_winner_id]

            else:
                self.last_winner_id = curr_output[-1]
                inferred_label = 0

            if self.verbose >= 2:
                print("Winner id:     ", self.last_winner_id)
                print("Inferred label:", inferred_label)

            # If this label is zero, that means this prototype is not labeled.
            if inferred_label == 0:
                # So, we give a pseudo label to the unknown winner.
                # These are negative temporary labels that is based on the id
                # of the prototype and generated as follows.
                self.proto_labels[self.last_winner_id] = -1 * (self.last_winner_id + 1)

                # So now this pseudo-label is our inferred label.
                inferred_label = self.proto_labels[self.last_winner_id]
                if self.verbose >= 1:
                    print("t=", self.time_step, "Allocated neuron", self.last_winner_id)


            
        # print("Pass 1")
        # Next we check if a user-provided label is available.
        if user_label != 0:
            # print("t=", self.time_step, "User label: ", user_label)
            # If so we need to access the most recent winner's label,
            # assuming the temporal causality between the prediction by the
            # system and the providence of the label;l by the user
            if self.verbose >= 2:
                print("time step:", int((self.time_step-1) % self.n_steps_per_sample)+1)
                print("user label:", user_label)
            if self.last_winner_id is not None:
                last_inferred_label = self.proto_labels[self.last_winner_id]

                # If the most recently predicted label (i.e. the one for the
                # current input which is also the user-provided label refer to)
                # is an actual label (not a pseudo one), then we check the
                # correctness of the predicted label against user-provided one.

                if last_inferred_label > 0:  # "Known Known class"
                    if last_inferred_label == user_label:
                        infer_check = 1
                        if self.verbose >= 2: print("Correct")
                    elif self.supervised == 1:
                        # If the error occurs, trigger allocation by sending an
                        # allocation signal
                        infer_check = -1
                        allocation_trigger = True
                        if self.verbose >= 2: print("Error")

                # If this prototype has a pseudo-label, then we label it with
                # the user-provided label and do not send any feedback (because
                # we did not have an actual prediction)

                elif last_inferred_label < 0:  # "Known Unknown class"
                    self.proto_labels[self.last_winner_id] = user_label
                    inferred_label = user_label
                    if self.verbose >= 1: print("New label ", user_label, " assigned to proto id ", self.last_winner_id)

            # There were more than one winner for sure during the last inference
            else:
                allocation_trigger = True

        # print("Pass 2")
        # Send out the readout predicted label (if any) and the feedback
        # about the correctness of this prediction after user providing the
        # actual label
        self.user_output.send(np.array([inferred_label]))
    
        # print("Pass 3")
        self.feedback.send(np.array([infer_check]))
        # print("Pass 4")

        if self.testing == 1:
            self.trigger_alloc.send(np.array([-1]))
        else:
            if allocation_trigger:
                self.trigger_alloc.send(np.array([1]))
            else:
                self.trigger_alloc.send(np.array([0]))
        # print("Pass 5")
        # print(inferred_label)
        # print("---------------------")
        


@implements(proc=Allocator, protocol=LoihiProtocol)
@requires(CPU)
@tag("fixed_pt")
class PyAllocatorModel(PyLoihiProcessModel):
    """Python implementation of the Allocator process.
    """

    trigger_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, np.int32)
    allocate_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, np.int32)
    next_alloc_id: np.int32 = LavaPyType(np.ndarray, np.int32)
    n_protos: np.int32 = LavaPyType(np.ndarray, np.int32)

    def __init__(self, proc_params):
        super().__init__(proc_params)

    def run_spk(self) -> None:
        # Allocation signal, initialized to a vector of zeros
        alloc_signal = np.zeros(shape=self.allocate_out.shape, dtype=np.int32)

        # Check the input, if a trigger for allocation is received then we
        # send allocation signal to the next neuron
        allocating = self.trigger_in.recv()[0]
        if allocating:
            # Choose the specific element of the OutPort to send allocate
            # signal. This is a single graded spike that has the payload of
            # the id of the next neuron to be allocated. Note that these id's
            # are starting from id=1, as the graded value of zero means no
            # signal. Hence, the initial value of next_alloc_id is one and
            # after each allocation it is incremented by one
            alloc_signal[0] = self.next_alloc_id

            # Increment this counter to point to the next neuron
            self.next_alloc_id += 1

        # Otherwise, just send zeros (i.e. no signal)
        self.allocate_out.send(alloc_signal)
