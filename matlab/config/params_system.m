% DC Motor Parameters
V_nom = 18.0; % V (Nominal input voltage)
tau_nom = 22.0e-3; % N*m (Nominal torque)
omega_nom = 3050 * (2 * pi / 60); % rad/s (Nominal speed)
I_nom = 0.540; % A (Nominal current)
R_m = 8.4; % Ohm (Terminal resistance)
k_t = 0.042; % N*m/A (Torque constant)
k_m = 0.042; % V/(rad/s) (Motor back-emf constant)
J_m = 4.0e-6; % kg*m^2 (Motor inertia)
L_m = 1.16e-3; % H (Motor inductance)

% Module Attachment Parameters
m_h = 0.0106; % kg (Module attachment hub mass)
r_h = 0.0111; % m (Module attachment hub radius)
J_h = 0.6e-6; % kg*m^2 (Module attachment moment of inertia)

% Inertia Disc Module
m_d = 0.053; % kg (Disc mass)
r_d = 0.0248; % m (Disc radius)

% Rotary Pendulum Module
m_r = 0.095; % kg (Rotary arm mass)
L_r = 0.085; % m (Rotary arm length from pivot to end of metal rod)
m_p = 0.024; % kg (Pendulum link mass)
L_p = 0.129; % m (Pendulum link length)

% Motor and Pendulum Encoders
encoder_line_count = 512; % lines/rev (Encoder line count)
encoder_quad_count = 2048; % lines/rev (Encoder line count in quadrature)
encoder_res_deg = 0.176; % deg/count (Encoder resolution in quadrature)
encoder_res_rad = 0.00307; % rad/count (Encoder resolution in quadrature)

% Amplifier Parameters
amplifier_type = 'PWM'; % (Amplifier type)
peak_current = 2.0; % A (Peak current)
continuous_current = 0.5; % A (Continuous current)
output_voltage_range_rec = 10; % V (Recommended output voltage range)
output_voltage_range_max = 15; % V (Maximum output voltage range)

% Gravity
g = 9.82; % m/s^2 

% Moment of Inertia 
Jp = 6e-5; 