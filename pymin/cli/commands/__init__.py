from collections import OrderedDict
from pymin.cli.commands.prune import PruneCommand

# from pymin.cli.commands.run_divide import RunDivideCommand
# from pymin.cli.commands.time_stability import TimeStabilityCommand
# from pymin.cli.commands.minimize import MinimizeCommand

WORKFLOW_COMMANDS = OrderedDict(
    [
        ("prune", PruneCommand),
        # ("run_divide", RunDivideCommand),
        # ("time_stability", TimeStabilityCommand),
        # ("minimize", MinimizeCommand),
    ]
)
