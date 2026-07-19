# ------------------------------------------------------------------------
# IP102 Distributed Continual Learning Runner for FOMO on Kaggle
# ------------------------------------------------------------------------

import os
import argparse
import subprocess
import pandas as pd
from tabulate import tabulate  # Kaggle has tabulate pre-installed, fallback to text print if not

def get_args_parser():
    parser = argparse.ArgumentParser('IP102 Distributed Runner', add_help=False)
    parser.add_argument('--model_name', default='google/owlvit-base-patch16', type=str,
                        help='Model name (e.g., google/owlvit-base-patch16 or google/owlvit-large-patch14)')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--num_workers', default=2, type=int,
                        help='Dataloader num_workers')
    parser.add_argument('--gpus', default=2, type=int,
                        help='Number of GPUs to use (Kaggle has 2x T4)')
    parser.add_argument('--output_dir', default='tmp/rwod', type=str,
                        help='Directory where results are saved')
    parser.add_argument('--tcp_port', default='29502', type=str,
                        help='TCP port for DDP coordination')
    return parser

def run_task(task_idx, prev_cls, curr_cls, args):
    print(f"\n=======================================================")
    print(f"               RUNNING TASK {task_idx}                ")
    print(f"=======================================================")
    
    # Configure image resize based on model choice
    image_resize = 840 if 'large' in args.model_name else 768

    # Base commands common to all tasks
    cmd = [
        "torchrun",
        f"--nproc_per_node={args.gpus}",
        "main.py",
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--dataset", "IP102",
        "--data_task", "OWOD",
        "--image_resize", str(image_resize),
        "--output_dir", args.output_dir,
        "--TCP", args.tcp_port,
        "--PREV_INTRODUCED_CLS", str(prev_cls),
        "--CUR_INTRODUCED_CLS", str(curr_cls),
        "--classnames_file", f"t{task_idx}_known.txt",
        "--prev_classnames_file", f"t{max(1, task_idx-1)}_known.txt" if task_idx > 1 else "t1_known.txt",
        "--test_set", "test.txt",
        "--train_set", "train.txt",
        "--unk_methods", "None",
        "--unk_method", "None",
        "--output_file", f"owod_ip102_t{task_idx}.csv"
    ]

    # For Tasks 1, 2, 3: specify ground-truth unknown classes file and use unknown proposals
    if task_idx < 4:
        cmd.extend([
            "--unknown_classnames_file", f"t{task_idx}_unknown_classnames_groundtruth.txt",
            "--unk_proposal"
        ])
    
    print(f"Executing command: {' '.join(cmd)}")
    
    # Run the subprocess and stream output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            
    rc = process.poll()
    if rc != 0:
        print(f"Error: Task {task_idx} exited with code {rc}")
        return False
    return True

def parse_and_display_results(args):
    print("\n=======================================================")
    print("               FINAL METRICS SUMMARY                   ")
    print("=======================================================")

    results = []
    
    # Metrics to extract
    metric_keys = {
        "K_AP50": "Known mAP",
        "PK_AP50": "Prev Known mAP",
        "CK_AP50": "Current Known mAP",
        "K_P50": "Known Precision",
        "K_R50": "Known Recall",
        "U_AP50": "Unknown mAP",
        "U_R50": "Unknown Recall"
    }

    for task_idx in range(1, 5):
        csv_path = os.path.join(args.output_dir, f"owod_ip102_t{task_idx}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: Result file for Task {task_idx} not found at {csv_path}")
            continue

        try:
            df = pd.read_csv(csv_path)
            # Take the last row in case of multiple evaluations in same CSV
            last_row = df.iloc[-1]
            
            task_res = {"Task": f"Task {task_idx}"}
            for key, display_name in metric_keys.items():
                if key in last_row:
                    val = last_row[key]
                    # Format as percentage or decimal
                    task_res[display_name] = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                else:
                    task_res[display_name] = "N/A"
                    
            results.append(task_res)
        except Exception as e:
            print(f"Error reading {csv_path}: {e}")

    if not results:
        print("No evaluation results were compiled.")
        return

    # Print results as a nice table
    headers = ["Task"] + list(metric_keys.values())
    table_data = [[res.get(h, "N/A") for h in headers] for res in results]
    
    try:
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    except ImportError:
        # Fallback to simple pandas print
        res_df = pd.DataFrame(results)
        print(res_df.to_string(index=False))

def main(args):
    # Tasks specs: (Task Index, Prev Cls, Cur Cls)
    tasks = [
        (1, 0, 7),
        (2, 7, 6),
        (3, 13, 6),
        (4, 19, 6)
    ]
    
    for task_idx, prev_cls, curr_cls in tasks:
        success = run_task(task_idx, prev_cls, curr_cls, args)
        if not success:
            print(f"Aborting runner due to failure in Task {task_idx}")
            return
            
    # Compile and print metrics
    parse_and_display_results(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('IP102 continual learning runner', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
