#!/usr/bin/env python3
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, required=True)
    args = parser.parse_args()

    N=s8=s4=stuck=0
    mins=[]; finals=[]; steps=[]
    with open(args.file,'r',encoding='utf-8') as fh:
        for line in fh:
            line=line.strip()
            if not line:
                continue
            d=json.loads(line)
            N+=1
            s8+=1 if d.get('success_8m') else 0
            s4+=1 if d.get('success_4m') else 0
            stuck+=1 if d.get('stuck') else 0
            mins.append(float(d.get('min_dist',0)))
            finals.append(float(d.get('final_dist',0)))
            steps.append(int(d.get('steps',0)))
    print('file', args.file)
    print('N', N)
    print('NavSucc@8m', s8)
    print('NavSucc@4m', s4)
    print('Stuck', stuck)
    print('avg_min_dist', (sum(mins)/len(mins)) if mins else None)
    print('avg_final_dist', (sum(finals)/len(finals)) if finals else None)
    print('avg_steps', (sum(steps)/len(steps)) if steps else None)

if __name__ == '__main__':
    main()