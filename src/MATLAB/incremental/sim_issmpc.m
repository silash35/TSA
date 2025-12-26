% Control of a distillation column subsystem (Alvarez et al, 2009) with MPC
%  basead on state-space model in the incremental form
clear
close all
clc
tic

% Ponto de operação em variavel de engenharia
% Usado como referência para definição de variáveis em desvio.
y_ref = [66.61255271 89.50113667 93.26343938]';
u_ref = [680 265 130 80]';

% MPC model
Atil = [ 0.37067676 0.03019531 0.01725641
      0.68652094 0.57037991 0.02130872
      2.46894109 1.32869029 0.0664796  ];

Btil = [ -0.00726326 -0.01558     0.00369239  0.02208961
       0.01693359  0.05265305  0.01081026 -0.07745152
      -0.0253843  -0.09184406  0.04534568  0.24170092 ];

Ctil = eye(3);

[A,B,C]=immpc(Atil,Btil,Ctil);

% Plant model
Ap=A;Bp=B;Cp=C;

% Updating the dimensions of variables
nu=size(B,2); % Number of manipulated inputs
ny=size(C,1); % Number of controlled outputs
nx=size(A,1); % Number of system states

% tuning parameters of MPC
p=120;% Output prediction horizon
m=3;% Input horizon
nsim=250;% Simulation time in sampling periods

q=[1,1,1]   ; % Output weights
r=[1,1,1,1] ; % Input moves weights
umax=[715 265 140 115]' - u_ref; % maximum value for inputs
umin=[600 187 130 80]' - u_ref; % minimum value for inputs
dumax=[99999 99999 99999 99999]'; % maximum variation for input moves

% Characteristics of process
uss = [0 0 0 0]';
yss = [0 0 0]';

%  Defining the initial conditions (deviation variables)
xmk=zeros(nx,1); % It starts the steady-state
xpk=xmk;
ypk=Cp*xpk;
uk_1=uss-uss;

% State observer
Kf = FKalman(ny,A,C,100);

% Starting simulation
ysp=[];
for in=1:nsim
    uk(:,in)=uk_1+uss;
    yk(:,in)=ypk+yss;

    if in <= 30
        ys=[0 0 0]'; % Set-point of the outputs
    else
        ys=[2.60473585 -8.24649468 8.04911466]';
    end

    [dukk,Vk]=issmpc(p,m,nu,ny,q,r,A,B,C,umax-uss,umin-uss,dumax,ys-yss,uk_1,xmk);
    duk=dukk(1:nu); % receding horizon
    Jk(in)=Vk; % control cost

    %Correction of the last control input
     xmk=A*xmk+B*duk;
     ymk=C*xmk;
  if in==100
      xpk=Ap*xpk+Bp*(duk+0.2*[1 1 1 1]'); % inserting unmeasured disturbance into plant
%       xpk=Ap*xpk+Bp*duk;
      ypk=Cp*xpk; % plant measurement
  else
      xpk=Ap*xpk+Bp*duk;
      ypk=Cp*xpk; % plant measurement
  end

  %Correction of the last measurement
  de=ypk-ymk;
  xmk=xmk+Kf*de;
  uk_1=duk+uk_1;
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
