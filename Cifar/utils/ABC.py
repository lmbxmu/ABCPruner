import sys
import random
import logging
import numpy as np
import math


#蜜源：解空间 当前层留多少百分比filter 1-100% 向量元素[0-1]
#适应函数：
#1)直接用原始精度的5/10个epoch
#2)只跑5/10个epoch
#cifar10 vgg

class BeeGroup(object):
	"""docstring for BeeGroup"""
	def __init__(self, arg):
		super(BeeGroup, self).__init__()
		self.code = [] #函数的维数 
		self.trueFit = sys.maxsize #记录真实的最小值 
		self.fitness = 0
		self.rfitness = 0 #相对适应值比例  
		self.trail = 0 #表示实验的次数，用于与limit作比较


global NP, FoodNumber, limit, maxCycle
global D, lb, ub
global result, NectraSource, EmployedBee, OnLooker
NP = 40 #种群的规模，采蜜蜂+观察蜂 
FoodNumber = 20 #食物的数量，为采蜜蜂的数量  
limit = 20 #限度，超过这个限度没有更新采蜜蜂变成侦查蜂
maxCycle = 10000 #停止条件  
D = 2 #函数的参数个数  
lb = -100 #函数的下界 
ub = 100 #函数的上界 
result = []
NectraSource = [] #蜜源
EmployedBee = [] #采蜜蜂
OnLooker = [] #观察蜂
BestSource = BeeGroup()



def initilize():
	for i in range(FoodNumber):
		NectraSource.append(BeeGroup())
		EmployedBee.append(BeeGroup())
		OnLooker.append(BeeGroup())
		for j in range(D):
			NectarSource[i].code[j]=random(lb,ub);  
            EmployedBee[i].code[j]=NectarSource[i].code[j];  
            OnLooker[i].code[j]=NectarSource[i].code[j];  
            BestSource.code[j]=NectarSource[0].code[j];  
        #初始化蜜源
        NectarSource[i].trueFit=calculationTruefit(NectarSource[i]);  
        NectarSource[i].fitness=calculationFitness(NectarSource[i].trueFit);  
        NectarSource[i].rfitness=0;  
        NectarSource[i].trail=0;  
        #采蜜蜂的初始化 
        EmployedBee[i].trueFit=NectarSource[i].trueFit;  
        EmployedBee[i].fitness=NectarSource[i].fitness;  
        EmployedBee[i].rfitness=NectarSource[i].rfitness;  
        EmployedBee[i].trail=NectarSource[i].trail;  
        #观察蜂的初始化  
        OnLooker[i].trueFit=NectarSource[i].trueFit;  
        OnLooker[i].fitness=NectarSource[i].fitness;  
        OnLooker[i].rfitness=NectarSource[i].rfitness;  
        OnLooker[i].trail=NectarSource[i].trail;   
        #最优蜜源的初始化  
        BestSource.trueFit=NectarSource[0].trueFit;  
        BestSource.fitness=NectarSource[0].fitness;  
        BestSource.rfitness=NectarSource[0].rfitness;  
        BestSource.trail=NectarSource[0].trail;  



def calculationTruefit(bee):
	truefit=0.5+(sin(sqrt(bee.code[0]*bee.code[0]+bee.code[1]*bee.code[1]))*sin(sqrt(bee.code[0]*bee.code[0]+bee.code[1]*bee.code[1]))-0.5)  
        /((1+0.001*(bee.code[0]*bee.code[0]+bee.code[1]*bee.code[1]))*(1+0.001*(bee.code[0]*bee.code[0]+bee.code[1]*bee.code[1])));  
  
    return truefit

def calculationFitness(truefit):
    if truefit >= 0:
    	return 1/(truefit + 1)
    else:
    	return 1 + abs(truefit)

def sendEmployedBees():
	for i in range(FoodNumber):
		param2change = np.random.random_integers(0,D-1)
	    while 1:
		    k = np.random.random_integers(0,FoodNumber-1)
		    if k != i:
		    	break
		for j in range(D):
			EmployedBee[i].code[j] = NectarSource[i].code[j]

		#采蜜蜂更新信息
		Rij = np.random.random_sample((-1,1))
		#根据公式(2-3)
		EmployedBee[i].code[param2change] = NectraSource[i].code[param2change]+Rij*(NectarSource[i].code[param2change]-NectarSource[k].code[param2change])
        if EmployedBee[i].code[param2change]>ub :
            EmployedBee[i].code[param2change]=ub
 
        if EmployedBee[i].code[param2change]<lb:  
            EmployedBee[i].code[param2change]=lb  
          
        EmployedBee[i].trueFit=calculationTruefit(EmployedBee[i]) 
        EmployedBee[i].fitness=calculationFitness(EmployedBee[i].trueFit) 
  
        #贪婪选择策略
        if EmployedBee[i].trueFit<NectarSource[i].trueFit:    
            for j in range(D):             
                NectarSource[i].code[j]=EmployedBee[i].code[j]              
            NectarSource[i].trail=0  
            NectarSource[i].trueFit=EmployedBee[i].trueFit
            NectarSource[i].fitness=EmployedBee[i].fitness 
        else:          
            NectarSource[i].trail = NectarSource[i].trail + 1
         
      


def CalculateProbabilities():#计算轮盘赌的选择概率
    maxfit=NectarSource[0].fitness  
    for i in range(1,FoodNumber):      
        if NectarSource[i].fitness>maxfit: 
            maxfit=NectarSource[i].fitness 
            
    for i in range(FoodNumber)      
        NectarSource[i].rfitness=(0.9*(NectarSource[i].fitness/maxfit))+0.1 
      

def sendOnlookerBees():#采蜜蜂与观察蜂交流信息，观察蜂更改信息 
    i=0;  
    t=0;  #是否超出食物源个数
    while t<FoodNumber:          
        R_choosed=np.random.random_sample((0,1))  
        if(R_choosed<NectarSource[i].rfitness)#根据被选择的概率选择  （算法搜索过程三的实现）              
            t++;  
            param2change=np.random.random_integers(0,D-1)               
            #选取不等于i的k 
            while 1:       
                k=param2change=np.random.random_integers(0,FoodNumber-1) 
                if (k!=i):             
                    break  
            for j in range(D):             
                OnLooker[i].code[j]=NectarSource[i].code[j]  
                          
            #更新 
            Rij=np.random.random_sample((-1,1))
            OnLooker[i].code[param2change]=NectarSource[i].code[param2change]+Rij*(NectarSource[i].code[param2change]-NectarSource[k].code[param2change])  
              
            #判断是否越界  
            if OnLooker[i].code[param2change]<lb:            
                OnLooker[i].code[param2change]=lb  
             
            if OnLooker[i].code[param2change]>ub:                  
                OnLooker[i].code[param2change]=ub  
              
            OnLooker[i].trueFit=calculationTruefit(OnLooker[i])  
            OnLooker[i].fitness=calculationFitness(OnLooker[i].trueFit) 
              
            #贪婪选择策略 
            if OnLooker[i].trueFit<NectarSource[i].trueFit:            
                for j in range(D):                    
                    NectarSource[i].code[j]=OnLooker[i].code[j]                   
                NectarSource[i].trail=0 
                NectarSource[i].trueFit=OnLooker[i].trueFit  
                NectarSource[i].fitness=OnLooker[i].fitness  
            else:              
                NectarSource[i].trail = NectarSource[i].trail + 1
     
        i = i + 1
        if i==FoodNumber:
            i=0 


#只有一只侦察蜂
def sendScoutBees():
	maxtrialindex = 0
	for i in range(1, FoodNumber):
		if NectraSource[i].trail > NectraSource[maxtrialindex].trail:
			maxtrialindex = i
	if NectraSource[maxtrialindex].trail >= limit:
		#重新初始化
		for j in range(D):
			R = np.random.random_sample((0,1))
			NectraSource[maxtrialindex].code[j] = lb + R*(ub-lb)
		NectraSource[maxtrialindex].trail = 0
		NectraSource[maxtrialindex].trueFit = calculationTruefit(NectraSource[maxtrialindex])
		NectraSource[maxtrialindex].fitness = calculationFitness(NectraSource[maxtrialindex].trueFit)

def MemorizeBestSource():#保留最优蜜源
	for i in range(1, FoodNumber):
		if NectarSource[i].trueFit < BestSource.trueFit:
			for j in range(D):
				BestSource.code[j] = NectarSource[i].code[j]
			BestSource.trueFit = NectarSource[i].trueFit


def main():

	setup_logging("./log.txt")
	logging.info("Begin ABC...")
	np.random.seed(0)
    #np.random.random_sample((min,max))
    initilize()
	MemorizeBestSource()
	for epoch in range(maxCycle):

        sendEmployedBees();  
              
        CalculateProbabilities();  
              
        sendOnlookerBees();  
              
        MemorizeBestSource();  
              
        sendScoutBees();  
              
        MemorizeBestSource();  
  
        logging.info(BestSource.trueFit)


if __name__ == '__main__':
    main()
