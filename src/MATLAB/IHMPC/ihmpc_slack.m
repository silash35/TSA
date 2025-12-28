clear all
close all
clc

% Ponto de operação em variavel de engenharia
% Usado como referência para definição de variáveis em desvio.
y_ref = [66.61255271 89.50113667 93.26343938]';
u_ref = [680 265 130 80]';

% MPC model
Ts = 1;
Atil = [ 0.37067676 0.03019531 0.01725641
    0.68652094 0.57037991 0.02130872
    2.46894109 1.32869029 0.0664796  ];

Btil = [ -0.00726326 -0.01558     0.00369239  0.02208961
    0.01693359  0.05265305  0.01081026 -0.07745152
    -0.0253843  -0.09184406  0.04534568  0.24170092 ];

Ctil = eye(3);
Dtil = zeros(size(Ctil,1), size(Btil,2));

sysd = ss(Atil, Btil, Ctil, Dtil, Ts);
Gz = tf(sysd);


[A,B,C,D0,Dd,F,Psi,N]=opom(Gz); % it creates opom model
Ap=A*1.00; Bp=B*1.05; Cp=C;
ny = size(C,1); % output variables of the system
nu = size(B,2); % input variables of the system
nx = size(A,1); % state variables of the system
nd = size(F,1); % states related to the stable pole of the system

% parameters of the IHMPC
sy = [1e3 1e3 1e3]; % weights of output slacks

nsim = 200;      % simulation time
m = 5;           % control horizon
qy = [1 1 1];      % output weights
qu = [];      % output weights
r  = [0.2 0.5 0.5 0.5]; % move weights

umax=[715 265 140 115]' - u_ref; % maximum value for inputs
umin=[600 187 130 80]' - u_ref; % minimum value for inputs
% =========================================================================

% Creating the Qy, Qybar and Rbar matrices
aux1=[];
aux2=[];
aux3=[];
Qy=diag(qy); % dimension (ny x ny)
S=diag(sy);
for i=1:m
    aux1=[aux1 qy];
    %     aux2=[aux2 qu];
    aux3=[aux3 r];
end
Qybar=diag(aux1); % dimension (m*ny x m*ny)

% Qubar=diag(aux2);
Rbar=diag(aux3);  % dimension (m*nu x m*nu)
Qaux = F'*Psi'*Qy*Psi*F ;
Qbar = dlyap(F,Qaux)    ; % dimension (nd x nd)
% Qaux = Psi'*Qy*Psi ;
% Qbar = dlyap(F,Qaux)    ; % dimension (nd x nd)
Bd=Dd*F*N;
Ibar=[];
Fx=[];
blin = zeros(ny,m*nu);
Fu = [];
dlin = blin;
Dm0=[];
Futil=[];
for j=1:m
    Ibar=[Ibar;eye(ny)]; % dimension (m*ny x ny)
    Fx=[Fx;Psi*(F^j)];   % dimension (m*ny x nd)
    blin = [Psi*(F^(j-1))*Bd blin(:,1:(m-1)*nu)];
    Fu = [Fu; blin]     ; % dimension (m*ny x m*nu)
    dlin = [D0 dlin(:,1:(m-1)*nu)];
    Dm0 = [Dm0; dlin]     ; % dimension (m*ny x m*nu)
    Futil=[Futil (F^(m-j))*Bd]; % dimension (nd x m*nu)
end
Dtil0 = Dm0((m-1)*ny+1:m*ny,:); % dimension (ny x m*nu)
Itil=[];
for k=1:m
    Itil = [Itil; eye(nu)]; % dimension m*nu x nu
end

aux=Itil;
Mtil=Itil;
for j=2:m
    aux=[zeros(nu);aux(1:(m-1)*nu,:)];
    Mtil=[Mtil aux]; % dimension m*nu x nu
end

% creating the H matrix of QP
H = [(Dm0+Fu)'*Qybar*(Dm0+Fu)+Futil'*Qbar*Futil+Rbar -(Dm0+Fu)'*Qybar*Ibar
    -Ibar'*Qybar*(Dm0+Fu)                           Ibar'*Qybar*Ibar+S];

% defining the initial conditions
% =========================================================================
u0 = [0 0 0 0]';
y0 = [0 0 0]';
ysp= [0 0 0]';

uk_1 = u0;

xmk = [y0; zeros(nd,1)]; % states of model
xpk = [y0; zeros(nd,1)]; % states of plant
ypk = Cp*xpk               ; % outputs of model
% =========================================================================
yspp=[];
for in=1:nsim
    ur(:,in) = uk_1  ;
    yr(:,in) = ypk   ;
    if in <= 20
        ysp    = [0 0 0]';
    elseif in <= 80
        ysp    = [1.33809828 -5.23812909 4.01255568]';
    elseif in <= 140
        ysp    = [0.48037208 -1.57549281  1.85025552]';
    else
        ysp    = [1.3533336 -2.90826548 4.64134915]';
    end
    cf = [(Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)'*Qybar*Dm0+...
        (Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)'*Qybar*Fu+(F^m*xmk(ny+1:end))'*Qbar*Futil ...
        (Ibar*ysp-Fx*xmk(ny+1:end)-Ibar*xmk(1:ny))'*Qybar*Ibar];
    c = (Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)'*Qybar*(Ibar*xmk(1:ny)+Fx*xmk(ny+1:end)-Ibar*ysp)+...
        (F^m*xmk(ny+1:end))'*Qbar*(F^m*xmk(ny+1:end));

    % Including inequality constraints
    Aineq = [ Mtil zeros(m*nu,ny)
        -Mtil zeros(m*nu,ny)];

    Bineq = [ Itil*(umax-uk_1)
        Itil*(uk_1-umin) ];

    % Including equality constraints
    Aeq = [Dtil0 -eye(ny)];

    Beq = ysp-xmk(1:ny);

    % options = optimset('display','iter')
    [dd,fvin,flagin]=quadprog(H,cf,Aineq,Bineq,Aeq,Beq);

    % storing calculated data
    fval(in) = dd'*H*dd + 2*cf*dd + c  ; % control cost value
    flag(in) = flagin                  ; % exitflag of exit condition of the MATLAB's quadprog routine
    duuk(:,in) = dd(1:m*nu)            ; % prediction of input moves
    sky(:,in)=dd(nu*m+1:m*nu+ny)       ; % slacks of outputs
    % ski(:,in)=dd(m*nu+ny+1:(m+1)*nu+ny); % slacks of inputs

    duk = dd(1:nu)   ; % receding horizon
    uk_1 = duk + uk_1; % inputs to be implemented in plant
    xpk = Ap*xpk+Bp*duk; % states of plant
    ypk = Cp*xpk      ; % output of plant
    xmk=xpk; % these should be computed through of state estimator (e.g Kalman filter)
    % ymk=C*xmk;
    yspp=[yspp ysp];
end

nc=size(yr,1);
figure(1)
for j=1:nc
    subplot(nc,1,j)
    plot(yr(j,:),'k-')
    hold on
    plot(yspp(j,:),'r--')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['y_' in];
    ylabel(yrot)
end

nc=size(ur,1);
figure(2)
for j=1:nc
    subplot(nc,1,j)
    plot(ur(j,:),'k-')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['u_' in];
    ylabel(yrot)
end

figure(3)
plot(fval)
xlabel('tempo nT')
ylabel('Cost function')

figure(4)
plot(flag)
xlabel('tempo nT')
ylabel('Flag')


nc=size(sky,1);
figure(5)
for j=1:nc
    subplot(nc,1,j)
    plot(sky(j,:),'k-')
    xlabel('tempo nT')
    in = num2str(j);
    yrot = ['\delta_' in];
    ylabel(yrot)
end

uk = ur;
yk = yr;
ysp = yspp;

save('output.mat', 'uk', 'yk', 'ysp');
