#!/usr/bin/python3.8
# coding=utf-8
"""Script for parsing and plotting arguments from VeriNet benchmark output. 

Part of bachelor thesis at BUT FIT.

Keyword arguments:
author -- David Hudák
login -- xhudak03
year -- 2022
"""

import argparse
import string
import re
import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

def return_status(line):
    if "Safe" in line:
        return "Safe"
    elif "Unsafe" in line:
        return "Unsafe"
    elif "Undecided" in line:
        return "Undecided"
    else:
        return "Dunno"


def parse_args():
    """
    Function parses arguments from command line and read VeriNet output logs.

    -f file -- file with VeriNet logs
    -c file -- optional, file with VeriNet logs for comparison.
               have to be same length as first file.

    Returns:
        Dictionary of lists with size of tested inputs.
    """
    parser = argparse.ArgumentParser(description='Parser of arguments.')
    parser.add_argument('-f', '--filename', dest='filename', metavar='f', type=argparse.FileType('r'),
                        help='Name of input file')
    parser.add_argument('-c', '--compare', dest='compare', metavar='c', type=argparse.FileType('r'),
                        help='Name of comparing input file')
    args = parser.parse_args()
    if args.filename == None:
        return None

    # Initializes dictionary

    experiments = {}
    experiments["Input"] = []
    experiments["Epsilon"] = []
    experiments["Time"] = []
    experiments["Result"] = []
    experiments["Branches"] = []
    experimentBool=False
    i = 0
    epsilon = 0

    # Reads logs of main VeriNet output.

    for line in args.filename.readlines():

        # Parses line from VeriNet log.

        if line.startswith("Benchmarking with epsilon ="):
            epsilon = re.findall(r'\d+\.*\d*', line)[0]
            continue
        elif experimentBool == False and line[0] != "F":
            continue
        elif experimentBool == False and line[0] == "F":
            experimentBool = True
        elif experimentBool == True and line[0] != "F":
            experimentBool = False
            continue
        reged = re.findall(r'\d+\.*\d*', line)

        experiments["Input"].append(reged[0])
        experiments["Epsilon"].append(epsilon)
        experiments["Time"].append(reged[-1])
        experiments["Branches"].append(reged[-3])
        experiments["Result"].append(return_status(line))

    # User called script with -c option. Same as lines before, but with different names.

    if args.compare != None:
        experiments["Input_comp"] = []
        experiments["Epsilon_comp"] = []
        experiments["Time_comp"] = []
        experiments["Result_comp"] = []
        experiments["Branches_comp"] = []
        i = 0
        epsilon = 0
        for line in args.compare.readlines():
            if line.startswith("Benchmarking with epsilon ="):
                epsilon = re.findall(r'\d+\.*\d*', line)[0]
                continue
            elif experimentBool == False and line[0] != "F":
                continue
            elif experimentBool == False and line[0] == "F":
                experimentBool = True
            elif experimentBool == True and line[0] != "F":
                experimentBool = False
                continue
            reged = re.findall(r'\d+\.*\d*', line)

            experiments["Input_comp"].append(reged[0])
            experiments["Epsilon_comp"].append(epsilon)
            experiments["Time_comp"].append(reged[-1])
            experiments["Branches_comp"].append(reged[-3])
            experiments["Result_comp"].append(return_status(line))

    return experiments
    
def plot_figures(pdf: pd.DataFrame):
    """
    Function plots figure from pandas dataframe. In final version not used.
    """
    print(pdf)
    pdf = pdf.astype({"Input": "int8", "Epsilon": "float", "Time": "float"})
    g = sns.relplot(data=pdf, x="Input", y="Time", col_wrap=2, hue="Result", col="Epsilon", ci=None, height=4)
    g.set(yscale="log")
    plt.savefig("plots.pdf")
    plt.show()

def print_table(pdf):
    """
    Function plots LaTeX tables used in our bachelor thesis.
    """
    contingency = pd.crosstab(pdf["Result"], pdf["Epsilon"])
    print(contingency.style.to_latex())
    contingency_times = pd.crosstab(pdf["Result"], columns=pdf["Epsilon"], values=pdf["Time"], aggfunc="sum", dropna=True)
    print(contingency_times.style.to_latex())
    contingency_branches = pd.crosstab(pdf["Result"], columns=pdf["Epsilon"], values=pdf["Branches"], aggfunc="mean", dropna=True)
    print(contingency_branches.style.to_latex())

def print_comp_branch_same(pdf):
    """
    Function plots LaTeX tables without cases which changed classification.
    """
    pdfE = pdf[pdf["Result"] == pdf["Result_comp"]]
    contingency_branches = pd.crosstab(pdfE["Result"], columns=pdfE["Epsilon"], values=pdfE["Branches"], aggfunc="mean", dropna=True)
    print(contingency_branches.style.to_latex())
    contingency_branches = pd.crosstab(pdfE["Result"], columns=pdfE["Epsilon"], values=pdfE["Branches_comp"], aggfunc="mean", dropna=True)
    print(contingency_branches.style.to_latex())

def print_comp_time_same(pdf):
    """
    Function plots LaTeX tables without cases which changed classification.
    """
    pdfE = pdf[pdf["Result"] == pdf["Result_comp"]]
    contingency_branches = pd.crosstab(pdfE["Result"], columns=pdfE["Epsilon"], values=pdfE["Time"], aggfunc="sum", dropna=True)
    print(contingency_branches.style.to_latex())
    contingency_branches = pd.crosstab(pdfE["Result"], columns=pdfE["Epsilon"], values=pdfE["Time_comp"], aggfunc="sum", dropna=True)
    print(contingency_branches.style.to_latex())

def print_comp_branch_not_same(pdf):
    """
    Function plots LaTeX tables without cases which did not change classification.
    """
    pdfEmptier = pdf[pdf["Result"] != pdf["Result_comp"]]
    # print(pdfEmptier.to_latex(columns=["Epsilon", "Time", "Time_comp", "Branches", "Branches_comp"]))
    contingency_branches = pd.crosstab(pdf["Result_comp"], columns=pdf["Epsilon"], values=pdf["Branches"], aggfunc="mean", dropna=True)
    print(contingency_branches.style.to_latex())
    contingency_branches = pd.crosstab(pdf["Result_comp"], columns=pdf["Epsilon"], values=pdf["Branches_comp"], aggfunc="mean", dropna=True)
    print(contingency_branches.style.to_latex())

def print_comp_time_not_same(pdf):
    """
    Function plots LaTeX tables without cases which did not change classification.
    """
    pdfE = pdf[pdf["Result"] != pdf["Result_comp"]]
    print(pdfE)
    contingency_branches = pd.crosstab(pdfE["Result"], columns=pdfE["Epsilon"], values=pdfE["Time"], aggfunc="sum", dropna=True)
    print(contingency_branches.style.to_latex())
    contingency_branches = pd.crosstab(pdfE["Result_comp"], columns=pdfE["Epsilon"], values=pdfE["Time_comp"], aggfunc="sum", dropna=True)
    print(contingency_branches.style.to_latex())


if __name__ == "__main__":
    experiments = parse_args()
    if experiments == None:
        print("No file has been choosen")
        exit(-1)
    pdf = pd.DataFrame(experiments)
    if "Input_comp" in experiments:
        pdf = pdf.astype({"Input": "int8", "Epsilon": "float", "Time": "float", 
                          "Branches": "int32", "Time_comp": "float", "Branches_comp": "int32"})
    else:
        pdf = pdf.astype({"Input": "int8", "Epsilon": "float", "Time": "float", 
                          "Branches": "int32"})
    # plot_figures(pdf)
    # print_table(pdf)
    # print_comp_branch_same(pdf)
    # print_comp_branch_not_same(pdf)
    print_comp_time_not_same(pdf)
    

