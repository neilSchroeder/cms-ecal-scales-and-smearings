import os 
import sys

#############################################################################################
def make_htcondor(out_dir, queue):

    lines = []
    lines.append( "executable = "+out_dir+"/htcondor.sh\n" )
    lines.append( "output = "+out_dir+"/"+os.path.basename(out_dir)+".out\n")
    lines.append( "error = "+out_dir+"/"+os.path.basename(out_dir)+".err\n")
    lines.append( "log = "+out_dir+"/"+os.path.basename(out_dir)+".log\n")
    lines.append( '+JobFlavour = "'+queue+'"\n')
    lines.append( "RequestCpus = 12\n")
    lines.append( "queue 1")

    htcondor = out_dir+"/htcondor"

    f = open(htcondor, "a")
    f.writelines(lines)
    f.close()
    return htcondor, out_dir+"/"+os.path.basename(out_dir)+"-done"

#############################################################################################
def make_script(cwd, cmd, out, script, done):

    lines = []
    lines.append("#!/bin/bash\n")
    lines.append("cd "+cwd+"\n")
    lines.append("eval `scramv1 runtime -sh`  uname -a\n")
    lines.append("echo $CMSSW_VERSION\n")
    lines.append('\n')
    lines.append(cmd+f" --from-condor\n")
    lines.append('\n')
    lines.append("touch "+done)

    f = open(script+".sh",'a')
    f.writelines(lines)
    f.close()

#############################################################################################
def manage(cmd, out, queue):
    
    cwd = os.getcwd()
    target_dir = cwd+"/condor/"+out

    #check for proper directory structure
    if not os.path.exists(target_dir):
        os.system("mkdir -p "+target_dir)
    else:
        print("[ERROR][python/condor_handler][manage] The directory {} already exists".format(target_dir))
        print("[INFO][python/condor_handler][manage] to continue this directory will be formatted.")
        cont = input("[INFO][python/condor_handler][manage] continue? [Y/n]:\n")
        if cont == 'Y':
            os.system("rm -rf "+target_dir)
            os.system("mkdir -p "+target_dir)
        elif cont == 'n':
            print("[EXIT]")
            return
        else:
            print("[ERROR][python/condor_handler][manage] could not parse input")
            print("[EXIT]")
            return

    htcondor, done = make_htcondor(target_dir, queue) 
    make_script(cwd, cmd, out, htcondor, done)

    condor_submit = "condor_submit --batch-name "+out+" "+htcondor
    print("[INFO][python/condor_handler][manage] submitting job to condor")
    print(condor_submit)
    os.system(condor_submit)
    return
