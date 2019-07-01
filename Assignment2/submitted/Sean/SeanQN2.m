% with hints
Yh = zeros(7,10);
n = 1;
for N = [25, 50, 75, 100, 150, 200, 300]
    s = 1;
    for seed = 1:10
        output = FORCE(1,30,2.5,N,true,seed,false);
        Yh(n,s) = output.nrmse;
        s =s +1;        
    end
    n = n+1;
end
%%
% without hints
Y = zeros(7,10);
n = 1;
for N = [25, 50, 75, 100, 150, 200, 300]
    s = 1;
    for seed = 1:10
        output = FORCE(1,30,2.5,N,false,seed,false);
        Y(n,s) = output.nrmse;
        s =s +1;        
    end
    n = n+1;
end
%% plotting

N = [25, 50, 75, 100, 150, 200, 300];
Yhm = mean(Yh,2);
Yhs = std(Yh,0,2);

Ym = mean(Y,2);
Ys = std(Y,0,2);

f=figure; 
clear a;
errorbar(N,Yhm,Yhs);
hold on
errorbar(N,Ym,Ys);
hold off
legend('with hints', 'w/o hints','Location','best')
xlabel('N')
ylabel('normalized error')

%%
%Qn: conclusion regarding hints during FORCE learning that they imply.
%Answer: Using "hints" reduces random error during learning and achieves
%lower error on average. 