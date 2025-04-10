import subprocess

subprocess.run('python step1a.py')
# train model, save model state

subprocess.run('python step1b.py')
# load latest state from step1a, generate proposals

subprocess.run('python step2.py')
# train RPN head using proposals from step1b

subprocess.run('python step3.py')
# load state from step2, train RPN head only

subprocess.run('python step4.py')
# load state from step3, train detection head only
