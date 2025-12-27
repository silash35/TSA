clear all
clc
s=tf('s');
nsim=80;
Ts=1;
g = [5/(10*s+1) 8/(8*s+1)    0
     0          3.3/(15*s+1) 7/(10*s+1)
     6/(15*s+1) 0            20/(20*s+1)];
% g = [-2.5/(9*s+1)   1/s 
%      -1.5/(7.5*s+1) 5.3/(10.2*s+1)]; 
% load('GF')
GF=g;
[t,ys] = step_response(GF,nsim);

[A,B,C,d0,Dd,F,psi,N]=opom(GF,Ts);

[ny,nu]=size(GF);
nx=size(A,1);
x0=zeros(nx,1);
passos=nsim/Ts;
for j=1:passos
    if j==1
        du=ones(nu,1);
    else
        du=zeros(nu,1);
    end
    xk=A*x0+B*du;
    y(:,j)=C*xk;
    x0=xk;
end

for j=1:ny
    figure(j)
    plot(Ts:Ts:Ts*passos,y(j,:),'ro',t,ys(:,j),'b-')
end