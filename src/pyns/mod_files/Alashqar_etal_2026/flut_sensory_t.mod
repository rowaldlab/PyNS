TITLE Sensory Axon Flut channels

: 01/26
: Abdallah Alashqar 
:
: FLUT segment channel properties for the tuned sensory axon model
:
: Fast K+ only
: Iterative equations H-H notation rest = -78.05 mV
:
: The tuned model is described in detail in:
:
: Alashqar A., Brihmat N., Gemar V., Hu Z., Koh S-R., Diaz-Pier S., de Freitas R.,
: Sasaki A., Milosevic M., Keesey R., Seanez I., Neufeld E., Cassoudesalle H., Hofstoetter U.,
: Minassian K., Wagner F., Rowald A., Virtual prototyping of non-invasive spinal cord 
: electrical stimulation targeting upper limb motor function, under review, 2026.
:
: 06/16
: Jessica Gaines
:
: Modification of channel properties
:
: 04/15
: Lane Heyboer
:
: Fast K+ current
: Ih current
:
: 02/02
: Cameron C. McIntyre
:
: Fast Na+, Persistant Na+, Slow K+, and Leakage currents 
: responsible for nodal action potential
: Iterative equations H-H notation rest = -80 mV
:
: This model is described in detail in:
: 
: Gaines JS, Finn KE, Slopsema JP, Heyboer LA, Polasek KH. A Model of 
: Motor and Sensory Axon Activation in the Median Nerve Using Surface 
: Electrical Stimulation. Journal of Computational Neuroscience, 2018.
:
: McIntyre CC, Richardson AG, and Grill WM. Modeling the excitability of
: mammalian nerve fibers: influence of afterpotentials on the recovery
: cycle. Journal of Neurophysiology 87:995-1006, 2002.

INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}

NEURON {
	SUFFIX flut_sensory_t
	NONSPECIFIC_CURRENT ikf
	RANGE gkf, ekf
    RANGE anA, anB, anC, bnA, bnB, bnC
	RANGE vtraub
	RANGE n_inf
	RANGE tau_n
}


UNITS {
	(mA) = (milliamp)
	(mV) = (millivolt)

}

PARAMETER {

    : conductance values
	gkf	= 0.02 (mho/cm2)

    : reversal potentials
	ekf	= -90.0 (mV)

    : read in from .hoc file
	celsius		(degC)
	dt              (ms)
	v               (mV)
	vtraub=-80	(mV)

    : parameters determining rate constants
    : fast K+
	anA = 0.0462	(1000/sec)
	anB = -83.2	(mV)
	anC = 1.1	(mV)
	bnA = 0.0824	(1000/sec)
	bnB = -60.8684	(mV)
	bnC = 10.5	(mV)
}

STATE {
	n
}

ASSIGNED {
	ikf	(mA/cm2)
	n_inf
	tau_n	(ms)
	q10_1
	q10_2
	q10_3
}

BREAKPOINT {
	SOLVE states METHOD cnexp
	ikf = gkf * n*n*n*n* (v-ekf)
}

DERIVATIVE states {   : exact Hodgkin-Huxley equations
       evaluate_fct(v)
	n' = (n_inf - n) / tau_n
}

UNITSOFF

INITIAL {
:
:	Q10 adjustment
:   Temperature dependence
:

	q10_1 = 2.2 ^ ((celsius-20)/ 10 )
	q10_2 = 2.9 ^ ((celsius-20)/ 10 )
	q10_3 = 3.0 ^ ((celsius-36)/ 10 )

	evaluate_fct(v)
	n = n_inf
}

PROCEDURE evaluate_fct(v(mV)) { LOCAL a,b,v2

	v2 = v - vtraub : convert to traub convention

    : fast K+
	a = q10_3*vtrapNA(v)
	b = q10_3*vtrapNB(v)
	tau_n = 1 / (a + b)
	n_inf = a / (a + b)
}

: vtrap functions to prevent discontinuity
FUNCTION vtrapNA(x){
    if(fabs((anB - x)/anC) < 1e-6){
        vtrapNA = anA*anC
    }else{
        vtrapNA = anA*(v-anB)/(1-Exp((anB-v)/anC))
    }
}

FUNCTION vtrapNB(x){
    if(fabs((x - bnB)/bnC) < 1e-6){
        vtrapNB = bnA*bnC  
    }else{
        vtrapNB = bnA*(bnB-v)/(1-Exp((v-bnB)/bnC))
    }
}

FUNCTION Exp(x) {
	if (x < -100) {
		Exp = 0
	}else{
		Exp = exp(x)
	}
}

UNITSON