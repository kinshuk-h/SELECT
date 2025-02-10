import pathlib
import argparse

import tqdm.auto as tqdm

from src.utils import common, io
from src.evaluation import dataset, task
from src.inference import ModelManager, sync_vram

MODELS = io.read_yaml('config/models.yaml')

INSTRUCTION = "Answer True or False: {concept_1} is an instance, subtype or category of {concept_2}"
GENERATE_KWARGS = dict(temperature=0.0001, do_sample=False, max_new_tokens=3)

def make_parser():
    parser = argparse.ArgumentParser(description="identifies how a model perceives concepts.")
    parser.add_argument("--data-dir", default="data", type=pathlib.Path,
                        help="root directory to load data from")
    parser.add_argument("-r", "--root-dir", default="results", type=pathlib.Path,
                        help="root directory to save generation results")
    parser.add_argument("-c", "--compose", action="store_true", help="run for compositions of concepts")
    parser.add_argument("-m", "--models", nargs="+", default=list(MODELS.keys()),
                        help="models to evaluate with, defaulting to the list defined in config/models.yaml")
    parser.add_argument("--backend", type=str, default='huggingface', choices=[ 'huggingface', 'openai', 'vllm' ],
                        help="backend to run inference with, defaulting to huggingface/openai.")
    return parser

def main():
    parser = make_parser()
    args   = parser.parse_args()

    suffix = '.compose' if args.compose else ''
    models = [ MODELS.get(model, model) for model in sorted(args.models or MODELS) ]

    dataset_state = dataset.DatasetState('select', compose=args.compose)

    level_map = {}
    def read_level(concept, _, level):
        level_map[concept] = level

    dataset_state.set_visitor(read_level)

    for model_name in models:
        print('> Using', model_name, 'for inference ...')

        batch_size = 32 * task.batch_multiplier(model_name)
        backend = 'openai' if 'openai' in model_name else backend
        if backend == 'vllm': batch_size = None

        pathsafe_model = io.pathsafe(model_name.lower().replace('/', '--'))
        results = io.Result(args.root_dir / f"expr.concept.understanding{suffix}" / f"{pathsafe_model}.json")

        instances = []

        # add positive examples corresponding to descendants
        for concept, node in dataset_state.items('node_data'):
            children = list(dataset.level_traverse(dict(XYZ=node))[1])
            children = [ (*concept[:-1], child) for child in children ]

            for child in children:
                if results.deepget(('#'.join(concept), '#'.join(child))): continue
                prompt = INSTRUCTION.format(
                    concept_1=dataset_state.node_data.deepget(child)['name'],
                    concept_2=node['name']
                )
                instances.append(dict(prompt=prompt, concept=(concept, child, True)))

        # add enegative examples corresponding to siblings
        for concept, siblings in dataset_state.items('sibling_map'):
            siblings = [ (*concept[:-1], sibling) for sibling in siblings ]

            for sibling in siblings:
                if results.deepget(('#'.join(concept), '#'.join(sibling))): continue
                prompt = INSTRUCTION.format(
                    concept_1=dataset_state.node_data.deepget(sibling)['name'],
                    concept_2=dataset_state.node_data.deepget(concept)['name']
                )
                instances.append(dict(prompt=prompt, concept=(concept, sibling, False)))

        if batch_size == None: batch_size = len(instances)
        batches = common.batchify(instances, batch_size=batch_size)

        with tqdm.tqdm(total=len(instances)) as pbar:
            timer   = common.BatchProgressTimer(pbar, len(batches))

            with ModelManager(model_name, backend=args.backend) as model:
                for idx, batch in enumerate(batches):
                    with timer.timed_operation(batch=idx+1):
                        prompts = model.make_prompt(batch['prompt'], chat=True)
                        outputs = model.generate(prompts, **GENERATE_KWARGS)

                        for (c1, c2, related), output in zip(batch['concept'], outputs):
                            results.deepset(('#'.join(c1), '#'.join(c2)), dict(
                                parent=c1, child=c2, delta=level_map[c2]-level_map[c1],
                                parent_level=level_map[c1], child_level=level_map[c2],
                                related=related, predicted=bool("true" in output.lower()),
                            ))
                            pbar.update()

                    results.save(); sync_vram()

if __name__ == "__main__":
    main()