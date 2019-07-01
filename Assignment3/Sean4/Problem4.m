load('Problem4_train.mat')
load('Problem4_test.mat')
% fdname can be 'quad', 'smooth', or 'exp'. 
fdname = 'quad';
[L,k,b]=fitGLM(Spikes_train,X_train,fdname);
Ltest=GLM_LL(Spikes_test,X_test,k,b,fdname);

imagesc(reshape(k,7,7))
