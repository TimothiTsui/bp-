% ��ջ�������
clc
clear
% 
%% ����ṹ����
%��ȡ����
info=readtable('price.csv');%��csv��ȡ����
infosize=height(info);

%%
input=[info.openPrice info.closePrice  info.lowestPrice info.highestPrice info.turnoverRate info.prechgPct];
input(infosize,:)=[];%���һ�е����벻��Ҫ
output=info.chgPct;
output(1,:)=[];%��һ�е��������Ҫ
%�������еĲ�������Ϊ��ҪԤ��ڶ�������ݣ�������ͬһ������� 

%%
%�ڵ����
inputnum=6;
hiddennum=5;
outputnum=1;

%ѵ�����ݺ�Ԥ������
input_train=input(2:5554,:)';
input_test=input(5555:5575,:)';
output_train=output(2:5554)';
output_test=output(5555:5575)';

%ѡ����������������ݹ�һ��
[inputn,inputps]=mapminmax(input_train);
[outputn,outputps]=mapminmax(output_train);

%��������
net=newff(inputn,outputn,hiddennum,{'logsig','purelin'},'trainrp');

%% �Ŵ��㷨������ʼ��
maxgen=500;                         %��������������������
sizepop=10;                        %��Ⱥ��ģ
pcross=0.4;                       %�������ѡ��0��1֮��
pmutation=0.2;                    %�������ѡ��0��1֮��

%�ڵ�����
numsum=inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum;

lenchrom=ones(1,numsum);        
bound=[-3*ones(numsum,1) 3*ones(numsum,1)];    %���ݷ�Χ

%------------------------------------------------------��Ⱥ��ʼ��--------------------------------------------------------
individuals=struct('fitness',zeros(1,sizepop), 'chrom',[]);  %����Ⱥ��Ϣ����Ϊһ���ṹ��
avgfitness=[];                      %ÿһ����Ⱥ��ƽ����Ӧ��
bestfitness=[];                     %ÿһ����Ⱥ�������Ӧ��
bestchrom=[];                       %��Ӧ����õ�Ⱦɫ��
%��ʼ����Ⱥ
for i=1:sizepop
    %�������һ����Ⱥ
    individuals.chrom(i,:)=Code(lenchrom,bound);    %���루binary��grey�ı�����Ϊһ��ʵ����float�ı�����Ϊһ��ʵ��������
    
    x=individuals.chrom(i,:);
    %������Ӧ��
    individuals.fitness(i)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   %Ⱦɫ�����Ӧ��
end

FitRecord=[];
%����õ�Ⱦɫ��
[bestfitness bestindex]=min(individuals.fitness);
bestchrom=individuals.chrom(bestindex,:);  %��õ�Ⱦɫ��
avgfitness=sum(individuals.fitness)/sizepop; %Ⱦɫ���ƽ����Ӧ��
%��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
trace=[avgfitness bestfitness]; 
%% ���������ѳ�ʼ��ֵ��Ȩֵ
% ������ʼ
for i=1:maxgen
    
    % ѡ��
    individuals=Select(individuals,sizepop); 
    %����
    individuals.chrom=Cross(pcross,lenchrom,individuals.chrom,sizepop,bound);
    % ����
    individuals.chrom=Mutation(pmutation,lenchrom,individuals.chrom,sizepop,i,maxgen,bound);
    
    % ������Ӧ�� 
    for j=1:sizepop
        x=individuals.chrom(j,:); %����
        individuals.fitness(j)=fun(x,inputnum,hiddennum,outputnum,net,inputn,outputn);   
    end
    
  %Ѱ������������
    [newbestfitness,newbestindex]=min(individuals.fitness);
    [worestfitness,worestindex]=max(individuals.fitness);
    
    % ������һ�ν�������õ�Ⱦɫ��
    if bestfitness>newbestfitness
        bestfitness=newbestfitness;
        bestchrom=individuals.chrom(newbestindex,:);
    end
    
    individuals.chrom(worestindex,:)=bestchrom;
    individuals.fitness(worestindex)=bestfitness;
    
    avgfitness=sum(individuals.fitness)/sizepop;
    trace=[trace; avgfitness bestfitness]; %��¼ÿһ����������õ���Ӧ�Ⱥ�ƽ����Ӧ��
    FitRecord=[FitRecord;individuals.fitness];
end

%% �����ų�ʼ��ֵȨֵ��������Ԥ��
w1=x(1:inputnum*hiddennum);
B1=x(inputnum*hiddennum+1:inputnum*hiddennum+hiddennum);
w2=x(inputnum*hiddennum+hiddennum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum);
B2=x(inputnum*hiddennum+hiddennum+hiddennum*outputnum+1:inputnum*hiddennum+hiddennum+hiddennum*outputnum+outputnum);

net.iw{1,1}=reshape(w1,hiddennum,inputnum);
net.lw{2,1}=reshape(w2,outputnum,hiddennum);
net.b{1}=reshape(B1,hiddennum,1);
net.b{2}=B2;

%% BP����ѵ��
%�����������
net.trainParam.epochs=100000;
net.trainParam.lr=0.1;
net.trainParam.goal=0.00001;
net.trainParam.max_fail=10000;

%����ѵ��
[net,per2]=train(net,inputn,outputn);

%% BP����Ԥ��
%���ݹ�һ��
inputn_test=mapminmax('apply',input_test,inputps);
an=sim(net,inputn_test);
BPoutput=mapminmax('reverse',an,outputps);

%����Ԥ����ͼ��
figure(1)
plot(BPoutput,':og')
hold on
plot(output_test,'-*');
legend('Ԥ���Ƿ�','ʵ���Ƿ�')
title('BP����Ԥ����ͨ�ͳ��Ƿ�','fontsize',12)
ylabel('�Ƿ�','fontsize',12)
xlabel('��������','fontsize',12)