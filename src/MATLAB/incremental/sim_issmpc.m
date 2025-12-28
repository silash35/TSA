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

ny = 3;
nu = 4;

Atil = [ 0.37067676 0.03019531 0.01725641
    0.68652094 0.57037991 0.02130872
    2.46894109 1.32869029 0.0664796  ];

Btil = [ -0.00726326 -0.01558     0.00369239  0.02208961
    0.01693359  0.05265305  0.01081026 -0.07745152
    -0.0253843  -0.09184406  0.04534568  0.24170092 ];

Ctil = eye(3);

[A,B,C]=immpc(Atil,Btil,Ctil);

Ap=A*1.00; Bp=B*1.05; Cp=C; % defining plant model

% Updating the dimensions of variables
nx=size(A,1); % Number of system states

% tuning parameters of MPC
p=120       ; % Output prediction horizon
m=5         ; % Input horizon
nsim=200    ; % Simulation time in sampling periods
q=[1,1,1]   ; % Output weights
r=[0.2 0.5 0.5 0.5] ; % Input moves weights

umax=[715 265 140 115]' - u_ref; % maximum value for inputs
umin=[600 187 130 80]' - u_ref; % minimum value for inputs
dumax=[99999 99999 99999 99999]'; % maximum variation for input moves

% Characteristics of process
uss = [0 0 0 0]'; % steady-state of the inputs
yss = [0 0 0]'; % steady-state of the outputs

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

    if in <= 20
        ys = [0; 0; 0];
    elseif in <= 80
        ys = [1.33809828; -5.23812909; 4.01255568];
    elseif in <= 140
        ys = [0.48037208; -1.57549281; 1.85025552];
    else
        ys = [1.3533336; -2.90826548; 4.64134915];
    end

    [dukk,Vk,flagin]=issmpc(p,m,nu,ny,q,r,A,B,C,umax-uss,umin-uss,dumax,ys-yss,uk_1,xmk);
    duk=dukk(1:nu); % receding horizon
    Jk(in)=Vk; % control cost
    flag(in)=flagin;

    %Correction of the last control input
     xmk=A*xmk+B*duk;
     ymk=C*xmk;

     xpk=Ap*xpk+Bp*duk;
     ypk=Cp*xpk; % plant measurement


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

figure(4)
plot(flag)
xlabel('tempo nT')

save('output.mat', 'uk', 'yk', 'ysp');
