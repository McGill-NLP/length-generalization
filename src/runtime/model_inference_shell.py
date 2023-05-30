import cmd
import sys
import time
from io import StringIO
from pathlib import Path
from pprint import pprint

import torch
from transformers import Seq2SeqTrainer, WEIGHTS_NAME


class Tee:
    def __init__(self):
        self.file = StringIO()
        self.stdout = sys.stdout

    def __del__(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()


class ModelInferenceShell(cmd.Cmd):
    intro = "Welcome to the model inference shell.   Type help or ? to list commands.\n"
    prompt = "(model) "

    def __init__(self, runtime, trainer: Seq2SeqTrainer, process_output_fn=None):
        super().__init__()
        self.io_obj = Tee()
        from runtime.seq2seq_runtime import Seq2SeqRuntime

        self.runtime: Seq2SeqRuntime = runtime
        self.trainer = trainer
        self.model = self.trainer.model

        self.max_length = self.trainer.args.generation_max_length
        self.num_beams = self.trainer.args.generation_num_beams

        self.target_tokens = None
        self.process_output = process_output_fn

    def do_load(self, checkpoint_name: str):
        """
        Load the checkpoint
        :param checkpoint_name: Name of the checkpoint, e.g. "checkpoint-10000"
        """
        try:
            ckpt_dir = self.load_ckpt(checkpoint_name)
            print(f"Checkpoint at {ckpt_dir} loaded.", file=self.io_obj)
        except Exception as exp:
            print("Couldn't load the checkpoint")
            print(exp, file=self.io_obj)

    def do_list_checkpoints(self, arg):
        """
        List all checkpoints
        """
        ckpt_dir: Path = self.runtime.exp_root / "checkpoints"
        if ckpt_dir.exists():
            for p in ckpt_dir.iterdir():
                if p.name.startswith("checkpoint-"):
                    print(p.name, file=self.io_obj)

    def do_set_gen_max_length(self, length):
        try:
            length = int(length)
            self.max_length = length
        except Exception as exp:
            print(exp, file=self.io_obj)

    def do_set_target_tokens(self, arg):
        """
        :param arg: e.g. ['1', '2', '3', '4', '5']
        :return:
        """
        target_tokens = eval(arg)
        self.target_tokens = [
            self.runtime.tokenizer.tokenize(tok)[0] for tok in target_tokens
        ]
        pprint(self.target_tokens, stream=self.io_obj)

    def do_generate(self, prompt: str):
        """
        Generate text from the model given a prompt
        :param prompt: str
        """
        prompt = eval(prompt)
        print(f"Prompt: ->{prompt}<-", file=self.io_obj)
        generate(
            self.runtime,
            self.model,
            prompt,
            max_length=self.max_length,
            num_beams=self.num_beams,
            process_output=self.process_output,
            io_obj=self.io_obj,
        )

    def do_next_token(self, prompt: str):
        """
        Generate the next token from a given prompt
        :param prompt:
        :return:
        """
        prompt = eval(prompt)
        print(f"Prompt: ->{prompt}<-", file=self.io_obj)
        predict_next_token(
            self.runtime,
            self.model,
            prompt,
            io_obj=self.io_obj,
            target_tokens=self.target_tokens,
        )

    def do_tokenize(self, prompt: str):
        """
        Return the tokenized prompt
        :param prompt:
        :return:
        """
        prompt = eval(prompt)
        pprint(self.runtime.tokenizer.tokenize(prompt), stream=self.io_obj)

    def do_history(self, arg):
        print("==================================", file=self.io_obj.stdout)
        print(self.io_obj.file.getvalue(), file=self.io_obj.stdout)
        print("==================================", file=self.io_obj.stdout)

    def do_save_shell(self, arg):
        """
        Save shell history (input/outputs) into filename
        :param filename:
        :return:
        """
        file_path = self.runtime.exp_root / f"shell_{time.time()}.txt"
        with file_path.open("w") as f:
            f.write(self.io_obj.file.getvalue())
        print(f"Saved history at {file_path}")

    def do_close(self, arg):
        return True

    def do_quit(self, arg):
        return True

    def do_exit(self, arg):
        return True

    def precmd(self, line):
        print(self.prompt, line, file=self.io_obj.file)
        try:
            return line
        except Exception as exp:
            print(exp, file=self.io_obj)

    def load_ckpt(self, name):
        ckpt_dir = self.runtime.exp_root / "checkpoints" / name
        state_dict = torch.load(ckpt_dir / WEIGHTS_NAME, map_location="cpu")
        self.trainer._load_state_dict_in_model(state_dict)
        self.model = self.trainer.model
        return ckpt_dir

def generate(
    runtime,
    model,
    prompt,
    max_length=None,
    num_beams=1,
    process_output=None,
    io_obj=None,
):
    inputs = runtime.tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        model = model.cuda()
    else:
        model = model

    model.eval()
    outputs = model.generate(
        inputs=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=max_length,
        num_beams=num_beams,
        pad_token_id=runtime.tokenizer.pad_token_id,
        eos_token_id=runtime.tokenizer.eos_token_id,
        return_dict_in_generate=True,
    )

    output_text = runtime.tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    if process_output is not None:
        output_text = process_output(output_text)

    print(output_text, file=io_obj)


def predict_next_token(
    runtime,
    model,
    prompt,
    io_obj=None,
    target_tokens=None,
):
    inputs = runtime.tokenizer(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
        model = model.cuda()
    else:
        model = model

    model.eval()
    outputs = model.generate(
        inputs=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=1,
        num_beams=1,
        pad_token_id=runtime.tokenizer.pad_token_id,
        eos_token_id=runtime.tokenizer.eos_token_id,
        return_dict_in_generate=True,
        do_sample=False,
        output_scores=True,
    )

    next_token_id = outputs.sequences[0, -1]
    next_token = runtime.tokenizer.convert_ids_to_tokens(int(next_token_id))

    output_text = runtime.tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(f"Next Token: {next_token}\n", file=io_obj)
    print(f"Output Text:", file=io_obj)
    print(output_text, file=io_obj)

    logits = outputs.scores[-1][0]
    probs = torch.softmax(logits, dim=-1)
    topk_tokens = torch.topk(probs, k=10)
    topk_tokens_probs = topk_tokens.values
    topk_tokens_ids = topk_tokens.indices

    print("\n\n", file=io_obj)
    print("Top 10 Predictions:", file=io_obj)
    for i, (tid, tp) in enumerate(zip(topk_tokens_ids, topk_tokens_probs)):
        tok = runtime.tokenizer.convert_ids_to_tokens(int(tid))
        print(f"{i + 1}:\t{tok}\tp: {tp}")

    if target_tokens is not None:
        target_token_indices = runtime.tokenizer.convert_tokens_to_ids(target_tokens)
        target_probs = probs[target_token_indices]
        topk_tokens = torch.topk(target_probs, k=len(target_token_indices))
        topk_tokens_probs = topk_tokens.values
        topk_tokens_indices = topk_tokens.indices

        print("\n\n", file=io_obj)
        print("Prediction on target tokens:", file=io_obj)
        for i, (idx, tp) in enumerate(zip(topk_tokens_indices, topk_tokens_probs)):
            tok = target_tokens[idx]
            print(f"{i + 1}:\t{tok}\tp: {tp}")
