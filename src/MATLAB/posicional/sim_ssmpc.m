clear
close all
clc
tic

% Ponto de operação em variavel de engenharia
% Usado como referência para definição de variáveis em desvio.
y_ref = [66.61255271 89.50113667 93.26343938]';
u_ref = [680 265 130 80]';

ny = 3;
nu = 4;

A = [ 0.37067676 0.03019531 0.01725641
      0.68652094 0.57037991 0.02130872
      2.46894109 1.32869029 0.0664796  ];

B = [ -0.00726326 -0.01558     0.00369239  0.02208961
       0.01693359  0.05265305  0.01081026 -0.07745152
      -0.0253843  -0.09184406  0.04534568  0.24170092 ];

C = eye(3);

Ap=A; Bp=B; Cp=C; % defining plant model

nx=size(A,1); % Number of system states
p=120       ; % Output prediction horizon
m=3         ; % Input horizon
nsim=250    ; % Simulation time in sampling periods
q=[1,1,1]   ; % Output weights
r=[1,1,1,1] ; % Input moves weights
umax=[715 265 140 115]' - u_ref; % maximum value for inputs
umin=[600 187 130 80]' - u_ref; % minimum value for inputs
dumax=[99999 99999 99999 99999]'; % maximum variation for input moves

uss = [0 0 0 0]'; % steady-state of the inputs
yss = [0 0 0]'; % steady-state of the outputs

% Setpoint
ysp = zeros(ny,nsim);
ys  = [2.60473585 -8.24649468 8.04911466]';% Set-point of the outputs
ysp(:,101:end) = repmat(ys, 1, nsim-100);

% Initial condition
u0 = [-1 2 1 0]';
y0 = [-2 2 3]';
[uk,yk,Jk]=ssmpc(p,m,nu,ny,nx,nsim,q,r,A,B,C,Ap,Bp,Cp,umax,umin,dumax,ys,uss,yss,y0,u0);

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

save('output.mat', 'uk', 'yk', 'ysp');

toc
