# Tools

these are tools to help you get somethings done, they may come in handy

## Add Uncertainties:

When the time comes to add uncertainties to your scales, this script will do the job for you.  
It takes as input a scales file produced by this framework and an uncertainty file built like a categories file (there are a couple examples in the main config folder).
It will output a scales file with the uncertainties you've provided included.

## Add MC Runs:

For scales and smearings files provided to EGamma there are MC runs that need to be included so that the proper uncertainties get applied and so that the various frameworks that use the scales files don't crash if they encounter something outside their expected run values.

This takes as input the scales file you want to add the runs to and an uncertainty file like the ones used in the script above to use the correct uncertainties for the mc bins.

## Scales Validator:

This is a useful script for checking that you haven't made a mistake in defining your categories. It will check that a scales file you give it has complete coverage in eta, R9, and Et (right now it only does eta and R9, Et is in the works).


