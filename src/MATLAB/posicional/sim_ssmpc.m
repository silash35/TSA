% Control of a distillation column subsystem (Alvarez et. al, 2009) with 
% MPC based on state-space model in the positional form
clear
close all
clc
tic
s = tf('s'); % defining the Laplace's variables
Ts = 1     ; % sampling time [=] min
g = [2.3/s         -0.7/s 
     4.7/(9.3*s +1) 1.4/(6.8*s+1)]; %  transfer function matrix
[ny,nu] = size(g);
% nu - Number of manipulated inputs
% ny - Number of controlled outputs
gd = c2d(g,Ts,'zoh');
[A,B,C,D] = ssdata(gd) ; % calcule the state-space model
% Ap=0.9*A; Bp=0.9*B; Cp=1.2*C; % defining plant model
% Ap=A; Bp=B; Cp=C; % defining plant model
gp = [2.3*1.01/s -0.7*1.02/s 
     1.1*4.7/(1.1*9.3*s +1) 1.05*1.4/(0.8*6.8*s+1)] ; % transfer function matrix
% gp=g;
gdp = c2d(gp,Ts,'zoh');
[Ap,Bp,Cp,Dp] = ssdata(gdp) ; % ca
% sysd = ss(A,B,C,D,Ts); % It is always necessary specify time sampling when discrete systems are considered here
% figure(2); step(sysd,80)
% hold on  ; step(gs,80)

nx=size(A,1); % Number of system states
p=120       ; % Output prediction horizon
% p=10       ; % Output prediction horizon
m=3         ; % Input horizon
nsim=250    ; % Simulation time in sampling periods
q=[0.5,1]     ; % Output weights
% q=[1,5e4];% Output weights
r=1*[1,1] ; % Input moves weights
umax=[10 10]'; % maximum value for inputs
umin=[0  0]' ; % minimum value for inputs
dumax=[1 1]'; % maximum variation for input moves

uss = [4.7 2.65]'; % steady-state of the inputs
yss = [47 52.5]'; % steady-state of the outputs
xss = calc_ss(Ap,Cp,nx,ny,yss); % steady-state of the states
ys  = [43 54]';% Set-point of the outputs
% ys=yss;

% Initial condition
u0 = [4 2]';
y0 = [40 50]';
x0 = calc_ss(Ap,Cp,nx,ny,y0);  
[uk,yk,Jk]=ssmpc(p,m,nu,ny,nx,nsim,q,r,A,B,C,Ap,Bp,Cp,umax,umin,dumax,ys,uss,yss,xss,y0,u0,x0);

ysp=[];
for i=1:nsim
    if i<= 100
        ys=yss;
    else
        ys=[43 54]';
    end
    ysp=[ysp ys];
end

nc=size(yk,1);
figure(1)
for j=1:nc
    subplot(nc,1,j)
    plot(yk(j,:),'k-')
    hold on
    plot(ysp(j,:),'r--')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['y_' in];
    ylabel(yrot)
end
% legend('PV','set-point')

nc=size(uk,1);
figure(2)
for j=1:nc
subplot(nc,1,j)
    plot(uk(j,:),'k-')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['u_' in];
    ylabel(yrot)
end

figure(3)
plot(Jk)
xlabel('tempo nT')
ylabel('Cost function')

% data print
% ======================================
% outputs
% ======================================
% y1 = [1:nsim;yk(1,:)]; 
% y2 = [1:nsim;yk(2,:)]; 
% ysp1= [1:nsim;ysp(1,:)]; 
% ysp2= [1:nsim;ysp(2,:)]; 
% 
% [fid,msg] = fopen('y1.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',y1)
% status = fclose(fid);
% 
% [fid,msg] = fopen('y2.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',y2)
% status = fclose(fid);
% 
% [fid,msg] = fopen('ysp1.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',ysp1)
% status = fclose(fid);
% 
% [fid,msg] = fopen('ysp2.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',ysp2)
% status = fclose(fid);
% 
% % ======================================
% 
% % ======================================
% % inputs
% % ======================================
% u1 = [1:nsim;uk(1,:)]; 
% u2 = [1:nsim;uk(2,:)];
% 
% [fid,msg] = fopen('u1.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',u1)
% status = fclose(fid);
% 
% [fid,msg] = fopen('u2.dat','w');
% fprintf(fid, '%6.3f  %6.3f\n',u2)
% status = fclose(fid);
% ======================================
toc
