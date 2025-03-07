import marimo

__generated_with = "0.11.14"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
        # 🐍 Cythonize Is All You Need

        ## Training a 1.5B parameter Cython coder with GRPO

        This is an example of how to use reinforcement learning to train a model to code in Cython using cythonize as feedback.
        """
    )
    return


@app.cell
def _(mo):
    # Define all the inputs
    oxen_repo_name = mo.ui.text(value="ox/Cython", full_width=True)
    output_oxen_repo_name = mo.ui.text(value="YOUR_USERNAME/REPO_NAME", full_width=True)
    model_name_ui = mo.ui.text(
        value="Qwen/Qwen2.5-Coder-1.5B-Instruct", full_width=True
    )
    train_file_ui = mo.ui.text(
        value="cython_test_passed_train.parquet", full_width=True
    )
    save_every = mo.ui.number(label="Save Every", value=100)
    commit_every = mo.ui.number(label="Commit Every", value=10)
    num_generations = mo.ui.number(label="Num Generations", value=4)
    use_peft_checkbox = mo.ui.checkbox(label="Use PEFT", value=True)
    use_gpu_checkbox = mo.ui.checkbox(label="Use GPU", value=True)

    run_form = (
        mo.md(
            """
        Base Model Name
        {model_name_ui}

        Dataset Repo Name
        {oxen_repo_name}

        Train Dataset (parquet)
        {train_file_ui}

        Output Repo Name
        {output_oxen_repo_name}

        {save_every}
        {commit_every}
        {num_generations}
        {use_peft_checkbox}
        {use_gpu_checkbox}
        """
        )
        .batch(
            oxen_repo_name=oxen_repo_name,
            output_oxen_repo_name=output_oxen_repo_name,
            model_name_ui=model_name_ui,
            train_file_ui=train_file_ui,
            save_every=save_every,
            commit_every=commit_every,
            num_generations=num_generations,
            use_peft_checkbox=use_peft_checkbox,
            use_gpu_checkbox=use_gpu_checkbox,
        )
        .form(
            submit_button_label="Train",
            bordered=False,
            show_clear_button=True,
            clear_button_label="Reset",
        )
    )

    run_form
    return (
        commit_every,
        model_name_ui,
        num_generations,
        output_oxen_repo_name,
        oxen_repo_name,
        run_form,
        save_every,
        train_file_ui,
        use_gpu_checkbox,
        use_peft_checkbox,
    )


@app.cell
def _(mo, oxen_repo_name, use_gpu_checkbox):
    mo.vstack(
        [
            mo.md(f"🐂 {oxen_repo_name.value}"),
            mo.md(f"Running experiment on gpu: {use_gpu_checkbox.value}"),
        ]
    )
    return


@app.cell
def _(
    AutoModelForCausalLM,
    AutoTokenizer,
    GRPOConfig,
    GRPOTrainer,
    LoraConfig,
    OxenExperiment,
    OxenTrainerCallback,
    RemoteRepo,
    CythonTool,
    SYSTEM_PROMPT,
    commit_every,
    create_dataset,
    extract_cython_code,
    extract_test_code,
    gc,
    get_peft_model,
    mo,
    model_name_ui,
    num_generations,
    os,
    output_oxen_repo_name,
    oxen_repo_name,
    response_contains_imports,
    response_contains_more_than_non_empty_line,
    response_contains_one_code_block,
    response_contains_one_test_block,
    run_form,
    save_every,
    setup_and_test_cython_project,
    torch,
    train_file_ui,
    use_gpu_checkbox,
    use_peft_checkbox,
):
    print(f"Checkbox val: {use_gpu_checkbox.value}")

    # If the button is not pressed, stop execution
    mo.stop(run_form.value is None)

    # Clean up memory between runs
    gc.collect()

    # Setup the Experiment
    model_name = model_name_ui.value

    output_dir = "outputs"
    repo = RemoteRepo(oxen_repo_name.value)
    train_dataset_file = train_file_ui.value
    if not os.path.exists(train_dataset_file):
        repo.download(train_dataset_file)

    output_repo = RemoteRepo(output_oxen_repo_name.value)
    experiment = OxenExperiment(output_repo, model_name, output_dir)
    train_dataset = create_dataset(train_dataset_file, SYSTEM_PROMPT)
    print(f"Running experiment in dir: {experiment.dir}")

    @experiment.log(f"non_empty_rewards.jsonl")
    def non_empty_reward_func(prompts, completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [response_contains_more_than_non_empty_line(c) for c in contents]

    @experiment.log(f"imports_rewards.jsonl")
    def imports_reward_func(prompts, completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [response_contains_imports(c) for c in contents]

    @experiment.log(f"test_block_count_rewards.jsonl")
    def test_block_count_reward_func(prompts, completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [response_contains_one_test_block(c) for c in contents]

    @experiment.log(f"code_block_count_rewards.jsonl")
    def code_block_count_reward_func(prompts, completions, **kwargs) -> list[float]:
        contents = [completion[0]["content"] for completion in completions]
        return [response_contains_one_code_block(c) for c in contents]

    @experiment.log(f"cythonize_compile_rewards.jsonl")
    def cythonize_compile_reward_func(prompts, completions, **kwargs) -> list[float]:
        # Extract the answers from the completions
        responses = [completion[0]["content"] for completion in completions]
        extracted_answers = [extract_cython_code(r) for r in responses]
        results = []
        for i, answer in enumerate(extracted_answers):
            data = {"cython_code": answer}
            tools = [CythonTool("compile")]
            cython_results = setup_and_test_cython_project(data, tools)
            score = 1.0 if cython_results["compile_passed"] else 0.0
            results.append(score)
        return results

    @experiment.log(f"cython_test_rewards.jsonl")
    def cython_test_reward_func(prompts, completions, **kwargs) -> list[float]:
        # Extract the answers from the completions
        responses = [completion[0]["content"] for completion in completions]
        extracted_codes = [extract_cython_code(r) for r in responses]
        extracted_tests = [extract_test_code(c) for c in extracted_codes]
        results = []
        for i, answer in enumerate(extracted_codes):
            score = 0.0
            if extracted_tests[i]:
                data = {"cython_code": answer}
                tools = [CythonTool("test")]
                cython_results = setup_and_test_cython_project(data, tools)
                # Let's give some extra credit for tests passing compared to the other rewards
                score = 3.0 if cython_results["test_passed"] else 0.0
            results.append(score)
        return results

    # Instantiate the model on the CPU or GPU
    device = "cpu"
    device_map = None
    attn_implementation = None
    if use_gpu_checkbox.value:
        device = "cuda"
        device_map = "auto"
        attn_implementation = "flash_attention_2"
    print(f"Using device: {device} -> '{use_gpu_checkbox.value}'")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        # attn_implementation=attn_implementation,
        device_map=device_map,
    ).to(device)

    peft_config = None
    if use_peft_checkbox.value:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=64,
            # Trying to get working with 16GB of VRAM
            target_modules="all-linear",
            # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            # target_modules=["q_proj", "k_proj", "v_proj"],
            task_type="CAUSAL_LM",
            lora_dropout=0.05,
        )
        model = get_peft_model(model, peft_config)

    model.enable_input_require_grads()

    with open(os.path.join(experiment.dir, "peft.txt"), "w") as f:
        trainable_params = 0
        all_param = 0
        for name, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                f.write(f"{name}\t{param.numel()}\n")  # Should show LoRA parameters
        param_str = f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        print(param_str)
        f.write(param_str)

    batch_size = 1
    num_examples = len(train_dataset)
    num_batches = num_examples / batch_size
    with mo.status.progress_bar(total=num_batches) as bar:
        training_args = GRPOConfig(
            output_dir=experiment.dir,
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            logging_steps=1,
            bf16=True,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=4,
            num_generations=num_generations.value,
            max_prompt_length=256,
            max_completion_length=786,
            num_train_epochs=1,
            save_steps=save_every.value,
            save_total_limit=1,
            max_grad_norm=0.1,
            log_on_each_node=False,
            optim="adamw_torch",
        )
        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[
                cythonize_compile_reward_func,  # 1.0 if cythonize compilation passes else 0.0
                cython_test_reward_func,  # 3.0 if passing tests else 0.0
                non_empty_reward_func,  # 1.0 if the code is not empty else 0.0
                test_block_count_reward_func,  # 1.0 if there is a test block else 0.0
                imports_reward_func,  # 1.0 if necessary imports are included
            ],
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,  # None if !use_peft_checkbox.value
            callbacks=[
                OxenTrainerCallback(experiment, bar, commit_every=commit_every.value)
            ],
        )
        trainer.train()
    return (
        all_param,
        attn_implementation,
        bar,
        batch_size,
        cythonize_compile_reward_func,
        cython_test_reward_func,
        code_block_count_reward_func,
        device,
        device_map,
        experiment,
        f,
        model,
        model_name,
        name,
        non_empty_reward_func,
        num_batches,
        num_examples,
        output_dir,
        output_repo,
        param,
        param_str,
        peft_config,
        repo,
        test_block_count_reward_func,
        imports_reward_func,
        tokenizer,
        train_dataset,
        train_dataset_file,
        trainable_params,
        trainer,
        training_args,
    )


@app.cell
def _(subprocess):
    class CythonTool:
        def __init__(self, name):
            self.name = name

        def run(self, results, project_dir):
            try:
                if self.name == "compile":
                    # Run cythonize to compile the module
                    result = subprocess.run(
                        ["python", "setup.py", "build_ext", "--inplace"],
                        cwd=project_dir,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )
                    results[f"compile_passed"] = result.returncode == 0
                    results[f"compile_stderr"] = str(result.stderr)
                elif self.name == "test":
                    # First ensure compilation
                    compile_result = subprocess.run(
                        ["python", "setup.py", "build_ext", "--inplace"],
                        cwd=project_dir,
                        capture_output=True,
                        text=True,
                        timeout=60,
                    )

                    if compile_result.returncode == 0:
                        # Run the test file
                        result = subprocess.run(
                            ["python", "test_module.py"],
                            cwd=project_dir,
                            capture_output=True,
                            text=True,
                            timeout=60,
                        )
                        results[f"test_passed"] = result.returncode == 0
                        results[f"test_stderr"] = str(result.stderr)
                    else:
                        results[f"test_passed"] = False
                        results[
                            f"test_stderr"
                        ] = "Compilation failed before testing: " + str(
                            compile_result.stderr
                        )
            except Exception as e:
                results[f"{self.name}_passed"] = False
                results[f"{self.name}_stderr"] = f"{e}"
            return results

    return (CythonTool,)


@app.cell
def _(Optional, re):
    def extract_regex(text: str, pattern: str) -> Optional[str]:
        # Use re.DOTALL to make '.' match newlines as well
        match = re.search(pattern, text, re.DOTALL)

        if match:
            return match.group(1)
        else:
            return None

    return (extract_regex,)


@app.cell
def _():
    def extract_code_regex():
        return r"```(?:cython|python|pyx)\n(.*?)\n```"

    return (extract_code_regex,)


@app.cell
def _():
    def extract_test_regex():
        return r"```python\n((?:.*?unittest.*?|.*?def\s+test_.*?\(.*?))\n```"

    return (extract_test_regex,)


@app.cell
def _(extract_code_regex, extract_regex):
    def extract_cython_code(response: str) -> str:
        code = extract_regex(response, extract_code_regex())
        if code:
            return code
        else:
            return response

    return (extract_cython_code,)


@app.cell
def _(extract_regex, extract_test_regex):
    def extract_test_code(response: str) -> str:
        return extract_regex(response, extract_test_regex())

    return (extract_test_code,)


@app.cell
def _(extract_cython_code):
    def response_contains_one_code_block(response: str) -> bool:
        # It has to have a ```cython``` or ```pyx``` block and a def or cdef
        code = extract_cython_code(response)
        if code and ("def " in code or "cdef " in code or "cpdef " in code):
            return 0.5
        else:
            return 0.0

    return (response_contains_one_code_block,)


@app.cell
def _(extract_test_code):
    def response_contains_one_test_block(response: str) -> bool:
        if extract_test_code(response):
            return 0.5
        else:
            return 0.0

    return (response_contains_one_test_block,)


@app.cell
def _(extract_cython_code):
    def response_contains_imports(response: str) -> bool:
        code = extract_cython_code(response)
        if not code:
            return 0.0

        # Check for essential Cython imports
        has_cython_import = False
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("import ") or line.startswith("from "):
                if "cython" in line.lower() or "numpy" in line.lower():
                    has_cython_import = True
                    break

        return 1.0 if has_cython_import else 0.0

    return (response_contains_imports,)


@app.cell
def _(
    extract_cython_code,
    response_contains_one_code_block,
    response_contains_one_test_block,
):
    def response_contains_more_than_non_empty_line(response: str) -> bool:
        if not (
            response_contains_one_code_block(response)
            and response_contains_one_test_block(response)
        ):
            return 0.0

        code = extract_cython_code(response)
        num_non_empty = 0
        for line in code.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                continue
            if len(line) < 2:
                continue
            num_non_empty += 1
        return 1.0 if num_non_empty >= 3 else 0.0

    return (response_contains_more_than_non_empty_line,)


@app.cell
def _():
    def template_pyx_file():
        return """# {code}
"""

    return (template_pyx_file,)


@app.cell
def _():
    def template_setup_py():
        return """
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules=cythonize("module.pyx"),
    include_dirs=[np.get_include()]
)
"""

    return (template_setup_py,)


@app.cell
def _():
    def template_test_file():
        return """
import unittest
from module import *

# {test_code}

if __name__ == "__main__":
    unittest.main()
"""

    return (template_test_file,)


@app.cell
def _(
    Path,
    extract_cython_code,
    extract_test_code,
    shutil,
    template_pyx_file,
    template_setup_py,
    template_test_file,
    uuid4,
):
    def setup_and_test_cython_project(row, tools):
        """
        Sets up a Cython project from template and runs tests for a single row of data
        """
        # Create temporary project directory
        project_dir = (
            Path("outputs") / Path("tests") / Path(f"temp_cython_project_{uuid4()}")
        )

        # mkdirs if they don't exist
        project_dir.mkdir(parents=True, exist_ok=True)

        # Read templates
        pyx_template = template_pyx_file()
        setup_template = template_setup_py()
        test_template = template_test_file()

        # Extract code and tests
        cython_code = extract_cython_code(row["cython_code"])
        test_code = extract_test_code(row["cython_code"]) or ""

        # Replace placeholders
        pyx_content = pyx_template.replace("# {code}", cython_code)
        test_content = test_template.replace("# {test_code}", test_code)

        # Write the project files
        module_path = project_dir / Path("module.pyx")
        with open(module_path, "w") as f:
            f.write(pyx_content)

        setup_path = project_dir / Path("setup.py")
        with open(setup_path, "w") as f:
            f.write(setup_template)

        test_path = project_dir / Path("test_module.py")
        with open(test_path, "w") as f:
            f.write(test_content)

        # Print the files for debugging
        print("Module.pyx content:")
        print(pyx_content)
        print("\nTest module content:")
        print(test_content)

        results = {}
        for tool in tools:
            results = tool.run(results, project_dir)

        for k, v in results.items():
            print("")
            print(k)
            print(v)

        print("=" * 80)

        # Clean up
        shutil.rmtree(project_dir)

        return results

    return (setup_and_test_cython_project,)


@app.cell
def _(OxenExperiment, TrainerCallback, Workspace, datetime, json, os):
    class OxenTrainerCallback(TrainerCallback):
        def __init__(self, experiment: OxenExperiment, progress_bar, commit_every):
            self.experiment = experiment
            self.bar = progress_bar
            self.commit_every = commit_every
            self.log_file_name = "logs.jsonl"
            self.log_file = os.path.join(self.experiment.dir, self.log_file_name)
            self.dst_dir = os.path.dirname(self.log_file)
            self.workspace = Workspace(
                experiment.repo,
                branch=f"{experiment.name}",
                workspace_name=f"training_run_{experiment.experiment_number}",
            )
            super().__init__()

        def on_log(self, args, state, control, logs=None, **kwargs):
            print("on_log.logs")
            # print(logs)

            # add timestamp to logs
            logs["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # save logs to file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(logs) + "\n")

        def on_step_end(self, args, state, control, **kwargs):
            print(f"on_step_end {state.global_step}")
            self.bar.update()

            if state.global_step % self.commit_every == 0:
                try:
                    for dir_path, _, files in os.walk(self.experiment.dir):
                        for file_name in files:
                            path = os.path.join(dir_path, file_name)
                            if path.endswith("jsonl"):
                                self.workspace.add(path, dst=str(self.experiment.dir))
                    self.workspace.commit(f"step {state.global_step} end GRPO")
                except Exception as e:
                    print(e)

    return (OxenTrainerCallback,)


@app.cell
def _(Any, Callable, Path, datetime, functools, json, os, time):
    class OxenExperiment:
        """
        An experiment helps log the experiment to an oxen repository,
        keeps track of the name and creates a corresponding branch to save results to
        """

        def __init__(self, repo, model_name, output_dir, experiment_type="GRPO"):
            self.repo = repo
            self.output_dir = output_dir

            branches = repo.branches()
            experiment_number = 0
            for branch in branches:
                if branch.name.startswith(f"{experiment_type}_"):
                    experiment_number += 1
            self.experiment_number = experiment_number
            short_model_name = model_name.split("/")[-1]
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.name = (
                f"{experiment_type}_{experiment_number}_{timestamp}_{short_model_name}"
            )
            self.dir = Path(os.path.join(self.output_dir, self.name))
            # Create the output directory if it doesn't exist
            os.makedirs(self.dir, exist_ok=True)

            print(f"Creating experiment branch {self.name}")
            repo.create_checkout_branch(self.name)

        def log(self, filename: str) -> Callable:
            """
            Create a decorator for a specific log file.

            Args:
                filename (str): Name of the log file to write to
            """
            log_path = self.dir / filename

            def decorator(func: Callable) -> Callable:
                @functools.wraps(func)
                def wrapper(*args, **kwargs) -> Any:
                    # Log the timestamp and function name
                    timestamp = datetime.now().isoformat()
                    func_name = func.__name__
                    start_time = time.time()
                    try:
                        # Execute the function
                        result = func(*args, **kwargs)

                        # Record one row for each of the results
                        for i, r in enumerate(result):
                            log_entry = {
                                "timestamp": timestamp,
                                "function": func_name,
                                "score": r,
                                "task_id": kwargs["task_id"][i],
                                "cython_prompt": kwargs["rust_prompt"][i],
                                "completion": kwargs["completions"][i][0]["content"],
                                "func_execution_time": time.time() - start_time,
                            }

                            # Write to log file
                            with open(log_path, "a") as f:
                                f.write(json.dumps(log_entry) + "\n")

                    except Exception as e:
                        print(f"Could not run func {func_name}: {e}")

                    return result

                return wrapper

            return decorator

    return (OxenExperiment,)


@app.cell
def _():
    def transform_dataset(system_prompt, dataset, tokenizer):
        data = dataset.map(
            lambda x: {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x["cython_prompt"]},
                    {"role": "assistant", "content": x["cython_code"]},
                ]
            }
        )
        tokenized_dataset = data.map(
            lambda x: tokenizer(
                [
                    tokenizer.apply_chat_template(conv, tokenize=False)
                    for conv in x["messages"]
                ],
                truncation=True,
                padding=True,
            ),
            batched=True,
            remove_columns=data.column_names,
        )
        return tokenized_dataset

    return (transform_dataset,)


@app.cell
def _(Dataset, load_dataset):
    def create_dataset(path, system_prompt) -> Dataset:
        data = load_dataset("parquet", data_files={"train": path})["train"]
        data = data.map(
            lambda x: {
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": x["cython_prompt"]},
                ],
                "test_list": x["cython_test_list"],
            }
        )
        print(data)
        return data

    return (create_dataset,)


@app.cell
def _():
    SYSTEM_PROMPT = """You are a pragmatic Cython programmer who enjoys test driven development. Given the following question, write a Cython function to complete the task. Make the code simple and easy to understand. The code should compile with cythonize. Try to make efficient use of Cython's type system to achieve good performance. When writing the function you can think through how to solve the problem and perform reasoning in the comments above the function.

    Then write Python unittest tests for the function you defined. Write multiple unit tests for the function. The tests should use Python's unittest framework with assertions. When writing the unit tests you can have comments specifying what you are testing in plain english.


    An example output should look like the following:

    ```cython
    # Reasoning goes here
    # and can be multi-line
    import numpy as np
    cimport numpy as np
    
    cpdef int add_nums(int x, int y):
        return x + y
    ```

    ```python
    import unittest
    from module import add_nums
    
    class TestAddNums(unittest.TestCase):
        def test_add_nums(self):
            # Test adding positive numbers
            self.assertEqual(add_nums(4, 2), 6)
            # Test adding a positive and negative number
            self.assertEqual(add_nums(4, -2), 2)
            # Test adding two negative numbers
            self.assertEqual(add_nums(-12, -1), -13)
    ```

    Make sure to respond with a Cython code block for the implementation and a Python code block for the tests. Include any necessary imports, especially for Cython and NumPy if you're working with arrays.
    """
    return (SYSTEM_PROMPT,)


@app.cell
def _():
    import marimo as mo
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer
    from peft import LoraConfig, get_peft_model
    from oxen import RemoteRepo, Workspace
    import os
    import re
    import numpy as np
    import json
    import torch
    from datetime import datetime
    import gc
    from pathlib import Path
    import functools
    from typing import Any, Callable, Optional
    from uuid import uuid4
    import subprocess
    import shutil
    import time

    return (
        Any,
        AutoModelForCausalLM,
        AutoTokenizer,
        Callable,
        Dataset,
        GRPOConfig,
        GRPOTrainer,
        LoraConfig,
        Optional,
        Path,
        RemoteRepo,
        TrainerCallback,
        Workspace,
        datetime,
        functools,
        gc,
        get_peft_model,
        json,
        load_dataset,
        mo,
        np,
        os,
        re,
        shutil,
        subprocess,
        time,
        torch,
        uuid4,
    )
