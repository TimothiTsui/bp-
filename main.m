% 清空环境变量
clc
clear
% 
%% 网络结构建立
%读取数据
info=readtable('price.csv');%从csv读取数据
infosize=height(info);

%%
input=[info.openPrice info.closePrice  info.lowestPrice info.highestPrice info.turnoverRate info.prechgPct];
input(infosize,:)=[];%最后一行的输入不需要
output=info.chgPct;
output(1,:)=[];%第一行的输出不需要
%上面三行的操作是因为，要预测第二天的数据，而不是同一天的数据 

%%
%节点个数
inputnum=6;
hiddennum=5;
outputnum=1;

%训练数据和预测数据
input_train=input(2:5554,:)';
input_test=input(5555:5575,:)';
output_train=output(2:5554)';
output_test=output(5555:5575)';

%选连样本输入输出数据归一化
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%构建网络
net=newff(inputn,outputn,hiddennum,{'logsig','purelin'},'trainrp');

%% 遗传算法参数初始化
maxgen=500;                         %进化代数，即迭代次数
sizepop=10;                        %种群规模
pcross=0.4;                       %交叉概率选择，0和1之间
pmutation=0.2;                    %变异概率选择，0和1之间

%节点总数
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

lenchrom=ones(1,numsum);        
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %数据范围

%------------------------------------------------------种群初始化--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %将种群信息定义为一个结构体
avgfitness=[];                      %每一代种群的平均适应度
bestfitness=[];                     %每一代种群的最佳适应度
bestchrom=[];                       %适应度最好的染色体
%初始化种群
for i=1:sizepop
    %随机产生一个种群
    individuals.chrom(i,:)=Code(lenchrom,bound);    %编码（binary和grey的编码结果为一个实数，float的编码结果为一个实数向量）
    
    x=individuals.chrom(i,:);
    %计算适应度
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %染色体的适应度
end

FitRecord=[];
%找最好的染色体
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %最好的染色体
avgfitness=sum(individuals.fitness)/sizepop; %染色体的平均适应度
%记录每一代进化中最好的适应度和平均适应度
trace=[avgfitness bestfitness]; 
%% 迭代求解最佳初始阀值和权值
% 进化开始
for i=1:maxgen
    
    % 选择
    individuals=Select(individuals,sizepop); 
    %交叉
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % 变异
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % 计算适应度 
    for j=1:sizepop
        x=individuals.chrom(j,:); %解码
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   
    end
    
  %寻找最优最差个体
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    
    % 代替上一次进化中最好的染色体
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/sizepop;
    trace=[trace; avgfitness bestfitness]; %记录每一代进化中最好的适应度和平均适应度
    FitRecord=[FitRecord;individuals.fitness];
end

%% 把最优初始阀值权值赋予网络预测
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP网络训练
%网络进化参数
net.trainParam.epochs=100000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.max_fail=10000;

%网络训练
[net,per2]=train(net,inputn,outputn);

%% BP网络预测
%数据归一化
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
BPoutput=mapminmax('reverse',an,outputps);

%网络预测结果图形
figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('预测涨幅','实际涨幅')
title('BP网络预测宇通客车涨幅','fontsize',12)
ylabel('涨幅','fontsize',12)
xlabel('测试样本','fontsize',12)