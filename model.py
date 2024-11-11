import gillespy2
import numpy as np
import pandas as pd
from dataclasses import dataclass


"""
Lets run the BMP2 Dimerization model

 1. BMP2 * Alk3 -> BMP2_Alk3                                            |  1r3on/off
 2. BMP2 * Alk8 -> BMP2_Alk8                                            |  1r8on/off
 3. BMP2 * RII -> BMP2_RII    											|  1rIIon/off
 4. BMP2_Alk3 * Alk3 -> BMP2_Alk3_Alk3 									|  2r3on/off
 5. BMP2_Alk3 * Alk8 -> BMP2_Alk3_Alk8									|  1r8on/off
 6. BMP2_Alk3 * RII -> BMP2_Alk3_RII									|  1rIIon/off
 7. BMP2_Alk8 * Alk3 -> BMP2_Alk3_Alk8									|  1r3on/off
 8. BMP2_Alk8 * Alk8 -> BMP2_Alk8_Alk8									|  2r8on/off
 9. BMP2_Alk8 * RII -> BMP2_Alk8_RII									|  1rIIon/off
10. BMP2_RII * Alk3 -> BMP2_Alk3_RII									|  1r3on/off
11. BMP2_RII * Alk8 -> BMP2_Alk8_RII									|  1r8on/off
12. BMP2_RII * RII -> BMP2_RII_RII  									|  2rIIon/off
13. BMP2_Alk8_Alk8 * RII -> BMP2_Alk8_Alk8_RII							|  1rIIon/off
14. BMP2_RII_RII * Alk3 -> BMP2_Alk3_RII_RII							|  1r3on/off
15. BMP2_RII_RII * Alk8 -> BMP2_Alk8_RII_RII							|  1r8on/off
16. BMP2_Alk3_Alk3 * RII -> BMP2_Alk3_Alk3_RII							|  1rIIon/off
17. BMP2_Alk3_Alk8 * RII -> BMP2_Alk3_Alk8_RII							|  1rIIon/off
18. BMP2_Alk3_RII * Alk3 -> BMP2_Alk3_Alk3_RII							|  2r3on/off
19. BMP2_Alk3_RII * Alk8 -> BMP2_Alk3_Alk8_RII							|  1r8on/off
20. BMP2_Alk3_RII * RII -> BMP2_Alk3_RII_RII							|  2rIIon/off
21. BMP2_Alk8_RII * Alk3 -> BMP2_Alk3_Alk8_RII							|  1r3on/off
22. BMP2_Alk8_RII * Alk8 -> BMP2_Alk8_Alk8_RII							|  2r8on/off
23. BMP2_Alk8_RII * RII -> BMP2_Alk8_RII_RII							|  2rIIon/off
24. BMP2_Alk3_Alk3_RII * RII -> BMP2_Alk3_Alk3_RII_RII					|  2rIIon/off
25. BMP2_Alk3_Alk8_RII * RII -> BMP2_Alk3_Alk8_RII_RII					|  2rIIon/off
26. BMP2_Alk3_RII_RII * Alk3 -> BMP2_Alk3_Alk3_RII_RII					|  2r3on/off
27. BMP2_Alk3_RII_RII * Alk8 -> BMP2_Alk3_Alk8_RII_RII					|  1r8on/off
28. BMP2_Alk8_Alk8_RII * RII -> BMP2_Alk8_Alk8_RII_RII					|  2rIIon/off
29. BMP2_Alk8_RII_RII * Alk3 -> BMP2_Alk3_Alk8_RII_RII					|  1r3on/off
30. BMP2_Alk8_RII_RII * Alk8 -> BMP2_Alk8_Alk8_RII_RII					|  2r8on/off


"""


@dataclass
class ParameterValues:
    timespan: np.ndarray
    A1: float
    init: pd.DataFrame


def SomeModel(parameter_values: ParameterValues | None = None) -> gillespy2.Model:

    # initialize
    model = gillespy2.Model(name="SSACSolver")
    init = parameter_values.init
    # parameters
    VOLUME = 2e-13
    STOCH = (1e9) / (6.022e23 * VOLUME)
    A2 = parameter_values.A1 / STOCH
    Boost_up = 50

    model.add_parameter(
        [
            k1A := gillespy2.Parameter(name="k1A", expression=0.0005 * STOCH * A2),
            k1r := gillespy2.Parameter(name="k1r", expression=(Ar3off := 0.0004)),
            k2A := gillespy2.Parameter(
                name="k2A", expression=0.0000011694 * STOCH * A2
            ),
            k2r := gillespy2.Parameter(name="k2r", expression=(Ar8off := 0.001197)),
            k3A := gillespy2.Parameter(name="k3", expression=0.0015 * STOCH * A2),
            k3r := gillespy2.Parameter(name="k3r", expression=(ArIIoff := 0.07)),
            k4 := gillespy2.Parameter(
                name="k4", expression=(Br3on := 0.0005 * STOCH) * Boost_up
            ),
            k4r := gillespy2.Parameter(name="k4r", expression=(Br3off := 0.0004)),
            k5 := gillespy2.Parameter(
                name="k5", expression=(Ar8on := 0.0000011694 * STOCH) * Boost_up
            ),
            k5r := gillespy2.Parameter(name="k5r", expression=Ar8off),
            k6 := gillespy2.Parameter(
                name="k6", expression=(ArIIon := 0.0015 * STOCH) * Boost_up
            ),
            k6r := gillespy2.Parameter(name="k6r", expression=ArIIoff),
            k7 := gillespy2.Parameter(
                name="k7", expression=(Ar3on := 0.0005 * STOCH) * Boost_up
            ),
            k7r := gillespy2.Parameter(name="k7r", expression=Ar3off),
            k8 := gillespy2.Parameter(
                name="k8", expression=(Br8on := 0.0000011694 * STOCH) * Boost_up
            ),
            k8r := gillespy2.Parameter(name="k8r", expression=(Br8off := 0.001197)),
            k9 := gillespy2.Parameter(name="k9", expression=ArIIon * Boost_up),
            k9r := gillespy2.Parameter(name="k9r", expression=ArIIoff),
            k10 := gillespy2.Parameter(name="k10", expression=Ar3on * Boost_up),
            k10r := gillespy2.Parameter(name="k10r", expression=Ar3off),
            k11 := gillespy2.Parameter(name="k11", expression=Ar8on * Boost_up),
            k11r := gillespy2.Parameter(name="k11r", expression=Ar8off),
            k12 := gillespy2.Parameter(
                name="k12", expression=(BrIIon := 0.0015 * STOCH) * Boost_up
            ),
            k12r := gillespy2.Parameter(name="k12r", expression=(BrIIoff := 0.07)),
            k13 := gillespy2.Parameter(name="k13", expression=ArIIon * Boost_up),
            k13r := gillespy2.Parameter(name="k13r", expression=ArIIoff),
            k14 := gillespy2.Parameter(name="k14", expression=Ar3on * Boost_up),
            k14r := gillespy2.Parameter(name="k14r", expression=Ar3off),
            k15 := gillespy2.Parameter(name="k15", expression=Ar8on * Boost_up),
            k15r := gillespy2.Parameter(name="k15r", expression=Ar8off),
            k16 := gillespy2.Parameter(name="k16", expression=ArIIon * Boost_up),
            k16r := gillespy2.Parameter(name="k16r", expression=ArIIoff),
            k17 := gillespy2.Parameter(name="k17", expression=ArIIon * Boost_up),
            k17r := gillespy2.Parameter(name="k17r", expression=ArIIoff),
            k18 := gillespy2.Parameter(name="k18", expression=Br3on * Boost_up),
            k18r := gillespy2.Parameter(name="k18r", expression=Br3off),
            k19 := gillespy2.Parameter(name="k19", expression=Ar8on * Boost_up),
            k19r := gillespy2.Parameter(name="k19r", expression=Ar8off),
            k20 := gillespy2.Parameter(name="k20", expression=BrIIon * Boost_up),
            k20r := gillespy2.Parameter(name="k20r", expression=BrIIoff),
            k21 := gillespy2.Parameter(name="k21", expression=Ar3on * Boost_up),
            k21r := gillespy2.Parameter(name="k21r", expression=Ar3off),
            k22 := gillespy2.Parameter(name="k22", expression=Br8on * Boost_up),
            k22r := gillespy2.Parameter(name="k22r", expression=Br8off),
            k23 := gillespy2.Parameter(name="k23", expression=BrIIon * Boost_up),
            k23r := gillespy2.Parameter(name="k23r", expression=BrIIoff),
            k24 := gillespy2.Parameter(name="k24", expression=BrIIon * Boost_up),
            k24r := gillespy2.Parameter(name="k24r", expression=BrIIoff),
            k25 := gillespy2.Parameter(name="k25", expression=BrIIon * Boost_up),
            k25r := gillespy2.Parameter(name="k25r", expression=BrIIoff),
            k26 := gillespy2.Parameter(name="k26", expression=Br3on * Boost_up),
            k26r := gillespy2.Parameter(name="k26r", expression=Br3off),
            k27 := gillespy2.Parameter(name="k27", expression=Ar8on * Boost_up),
            k27r := gillespy2.Parameter(name="k27r", expression=Ar8off),
            k28 := gillespy2.Parameter(name="k28", expression=BrIIon * Boost_up),
            k28r := gillespy2.Parameter(name="k28r", expression=BrIIoff),
            k29 := gillespy2.Parameter(name="k29", expression=Ar3on * Boost_up),
            k29r := gillespy2.Parameter(name="k29r", expression=Ar3off),
            k30 := gillespy2.Parameter(name="k30", expression=Br8on * Boost_up),
            k30r := gillespy2.Parameter(name="k30r", expression=Br8off),
            k31A := gillespy2.Parameter(name="k31A", expression=0.00014 * STOCH * A2),
            k31r := gillespy2.Parameter(name="k31r", expression=(A7r3off := 0.0079)),
            k32A := gillespy2.Parameter(
                name="k32A", expression=0.0000023388 * STOCH * A2
            ),
            k32r := gillespy2.Parameter(name="k32r", expression=(A7r8off := 0.001197)),
            k33A := gillespy2.Parameter(name="k33", expression=0.0015 * STOCH * A2),
            k33r := gillespy2.Parameter(name="k33r", expression=(A7rIIoff := 0.009)),
            k34 := gillespy2.Parameter(
                name="k34", expression=(B7r3on := 0.00014 * STOCH) * Boost_up
            ),
            k34r := gillespy2.Parameter(name="k34r", expression=(B7r3off := 0.0079)),
            k35 := gillespy2.Parameter(
                name="k35", expression=(A7r8on := 0.000002338 * STOCH) * Boost_up
            ),
            k35r := gillespy2.Parameter(name="k35r", expression=A7r8off),
            k36 := gillespy2.Parameter(
                name="k36", expression=(A7rIIon := 0.0014 * STOCH) * Boost_up
            ),
            k36r := gillespy2.Parameter(name="k36r", expression=A7rIIoff),
            k37 := gillespy2.Parameter(
                name="k37", expression=(A7r3on := 0.00014 * STOCH) * Boost_up
            ),
            k37r := gillespy2.Parameter(name="k37r", expression=A7r3off),
            k38 := gillespy2.Parameter(
                name="k38", expression=(B7r8on := 0.000002338 * STOCH) * Boost_up
            ),
            k38r := gillespy2.Parameter(name="k38r", expression=(B7r8off := 0.001197)),
            k39 := gillespy2.Parameter(name="k39", expression=A7rIIon * Boost_up),
            k39r := gillespy2.Parameter(name="k39r", expression=A7rIIoff),
            k40 := gillespy2.Parameter(name="k40", expression=A7r3on * Boost_up),
            k40r := gillespy2.Parameter(name="k40r", expression=A7r3off),
            k41 := gillespy2.Parameter(name="k41", expression=A7r8on * Boost_up),
            k41r := gillespy2.Parameter(name="k41r", expression=A7r8off),
            k42 := gillespy2.Parameter(
                name="k42", expression=(B7rIIon := 0.0014 * STOCH) * Boost_up
            ),
            k42r := gillespy2.Parameter(name="k42r", expression=(B7rIIoff := 0.009)),
            k43 := gillespy2.Parameter(name="k43", expression=A7rIIon * Boost_up),
            k43r := gillespy2.Parameter(name="k43r", expression=A7rIIoff),
            k44 := gillespy2.Parameter(name="k44", expression=A7r3on * Boost_up),
            k44r := gillespy2.Parameter(name="k44r", expression=A7r3off),
            k45 := gillespy2.Parameter(name="k45", expression=A7r8on * Boost_up),
            k45r := gillespy2.Parameter(name="k45r", expression=A7r8off),
            k46 := gillespy2.Parameter(name="k46", expression=A7rIIon * Boost_up),
            k46r := gillespy2.Parameter(name="k46r", expression=A7rIIoff),
            k47 := gillespy2.Parameter(name="k47", expression=A7rIIon * Boost_up),
            k47r := gillespy2.Parameter(name="k47r", expression=A7rIIoff),
            k48 := gillespy2.Parameter(name="k48", expression=B7r3on * Boost_up),
            k48r := gillespy2.Parameter(name="k48r", expression=B7r3off),
            k49 := gillespy2.Parameter(name="k49", expression=A7r8on * Boost_up),
            k49r := gillespy2.Parameter(name="k49r", expression=A7r8off),
            k50 := gillespy2.Parameter(name="k50", expression=B7rIIon * Boost_up),
            k50r := gillespy2.Parameter(name="k50r", expression=B7rIIoff),
            k51 := gillespy2.Parameter(name="k51", expression=A7r3on * Boost_up),
            k51r := gillespy2.Parameter(name="k51r", expression=A7r3off),
            k52 := gillespy2.Parameter(name="k52", expression=B7r8on * Boost_up),
            k52r := gillespy2.Parameter(name="k52r", expression=B7r8off),
            k53 := gillespy2.Parameter(name="k53", expression=B7rIIon * Boost_up),
            k53r := gillespy2.Parameter(name="k53r", expression=B7rIIoff),
            k54 := gillespy2.Parameter(name="k54", expression=B7rIIon * Boost_up),
            k54r := gillespy2.Parameter(name="k54r", expression=B7rIIoff),
            k55 := gillespy2.Parameter(name="k55", expression=B7rIIon * Boost_up),
            k55r := gillespy2.Parameter(name="k55r", expression=B7rIIoff),
            k56 := gillespy2.Parameter(name="k56", expression=B7r3on * Boost_up),
            k56r := gillespy2.Parameter(name="k56r", expression=B7r3off),
            k57 := gillespy2.Parameter(name="k57", expression=A7r8on * Boost_up),
            k57r := gillespy2.Parameter(name="k57r", expression=A7r8off),
            k58 := gillespy2.Parameter(name="k58", expression=B7rIIon * Boost_up),
            k58r := gillespy2.Parameter(name="k58r", expression=B7rIIoff),
            k59 := gillespy2.Parameter(name="k59", expression=A7r3on * Boost_up),
            k59r := gillespy2.Parameter(name="k59r", expression=A7r3off),
            k60 := gillespy2.Parameter(name="k60", expression=B7r8on * Boost_up),
            k60r := gillespy2.Parameter(name="k60r", expression=B7r8off),
            k61A := gillespy2.Parameter(name="k61A", expression=0.0005 * STOCH * A2),
            k61r := gillespy2.Parameter(name="k61r", expression=(A27r3off := 0.0004)),
            k62A := gillespy2.Parameter(
                name="k62A", expression=0.0000023388 * STOCH * A2
            ),
            k62r := gillespy2.Parameter(name="k62r", expression=(A27r8off := 0.001197)),
            k63A := gillespy2.Parameter(name="k63", expression=0.0014 * STOCH * A2),
            k63r := gillespy2.Parameter(name="k63r", expression=(A27rIIoff := 0.009)),
            k64 := gillespy2.Parameter(
                name="k64", expression=(B27r3on := 0.00014 * STOCH) * Boost_up
            ),
            k64r := gillespy2.Parameter(name="k64r", expression=(B27r3off := 0.0079)),
            k65 := gillespy2.Parameter(
                name="k65", expression=(A27r8on := 0.000002338 * STOCH) * Boost_up
            ),
            k65r := gillespy2.Parameter(name="k65r", expression=A27r8off),
            k66 := gillespy2.Parameter(
                name="k66", expression=(A27rIIon := 0.0014 * STOCH) * Boost_up
            ),
            k66r := gillespy2.Parameter(name="k66r", expression=A27rIIoff),
            k67 := gillespy2.Parameter(
                name="k67", expression=(A27r3on := 0.0005 * STOCH) * Boost_up
            ),
            k67r := gillespy2.Parameter(name="k67r", expression=A27r3off),
            k68 := gillespy2.Parameter(
                name="k68", expression=(B27r8on := 0.0000011694 * STOCH) * Boost_up
            ),
            k68r := gillespy2.Parameter(name="k68r", expression=(B27r8off := 0.001197)),
            k69 := gillespy2.Parameter(name="k69", expression=A27rIIon * Boost_up),
            k69r := gillespy2.Parameter(name="k69r", expression=A27rIIoff),
            k70 := gillespy2.Parameter(name="k70", expression=A27r3on * Boost_up),
            k70r := gillespy2.Parameter(name="k70r", expression=A27r3off),
            k71 := gillespy2.Parameter(name="k71", expression=A27r8on * Boost_up),
            k71r := gillespy2.Parameter(name="k71r", expression=A27r8off),
            k72 := gillespy2.Parameter(
                name="k72", expression=(B27rIIon := 0.0015 * STOCH) * Boost_up
            ),
            k72r := gillespy2.Parameter(name="k72r", expression=(B27rIIoff := 0.07)),
            k73 := gillespy2.Parameter(name="k73", expression=A27rIIon * Boost_up),
            k73r := gillespy2.Parameter(name="k73r", expression=A27rIIoff),
            k74 := gillespy2.Parameter(name="k74", expression=A27r3on * Boost_up),
            k74r := gillespy2.Parameter(name="k74r", expression=A27r3off),
            k75 := gillespy2.Parameter(name="k75", expression=A27r8on * Boost_up),
            k75r := gillespy2.Parameter(name="k75r", expression=A27r8off),
            k76 := gillespy2.Parameter(name="k76", expression=A27rIIon * Boost_up),
            k76r := gillespy2.Parameter(name="k76r", expression=A27rIIoff),
            k77 := gillespy2.Parameter(name="k77", expression=A27rIIon * Boost_up),
            k77r := gillespy2.Parameter(name="k77r", expression=A27rIIoff),
            k78 := gillespy2.Parameter(name="k78", expression=B27r3on * Boost_up),
            k78r := gillespy2.Parameter(name="k78r", expression=B27r3off),
            k79 := gillespy2.Parameter(name="k79", expression=A27r8on * Boost_up),
            k79r := gillespy2.Parameter(name="k79r", expression=A27r8off),
            k80 := gillespy2.Parameter(name="k80", expression=B27rIIon * Boost_up),
            k80r := gillespy2.Parameter(name="k80r", expression=B27rIIoff),
            k81 := gillespy2.Parameter(name="k81", expression=A27r3on * Boost_up),
            k81r := gillespy2.Parameter(name="k81r", expression=A27r3off),
            k82 := gillespy2.Parameter(name="k82", expression=B27r8on * Boost_up),
            k82r := gillespy2.Parameter(name="k82r", expression=B27r8off),
            k83 := gillespy2.Parameter(name="k83", expression=B27rIIon * Boost_up),
            k83r := gillespy2.Parameter(name="k83r", expression=B27rIIoff),
            k84 := gillespy2.Parameter(name="k84", expression=B27rIIon * Boost_up),
            k84r := gillespy2.Parameter(name="k84r", expression=B27rIIoff),
            k85 := gillespy2.Parameter(name="k85", expression=B27rIIon * Boost_up),
            k85r := gillespy2.Parameter(name="k85r", expression=B27rIIoff),
            k86 := gillespy2.Parameter(name="k86", expression=B27r3on * Boost_up),
            k86r := gillespy2.Parameter(name="k86r", expression=B27r3off),
            k87 := gillespy2.Parameter(name="k87", expression=A27r8on * Boost_up),
            k87r := gillespy2.Parameter(name="k87r", expression=A27r8off),
            k88 := gillespy2.Parameter(name="k88", expression=B27rIIon * Boost_up),
            k88r := gillespy2.Parameter(name="k88r", expression=B27rIIoff),
            k89 := gillespy2.Parameter(name="k89", expression=A27r3on * Boost_up),
            k89r := gillespy2.Parameter(name="k89r", expression=A27r3off),
            k90 := gillespy2.Parameter(name="k90", expression=B27r8on * Boost_up),
            k90r := gillespy2.Parameter(name="k90r", expression=B27r8off),
            k1000 := gillespy2.Parameter(name="k1000", expression=0.0005),
            k2000 := gillespy2.Parameter(name="k2000", expression=0.0005),
            k7000 := gillespy2.Parameter(name="k7000", expression=0.0005),
        ]
    )

    # Species
    model.add_species(
        [
            Alk3 := gillespy2.Species(
                name="Alk3", initial_value=int(init["Alk3"].iloc[-1])
            ),
            Alk8 := gillespy2.Species(
                name="Alk8", initial_value=int(init["Alk8"].iloc[-1])
            ),
            RII := gillespy2.Species(
                name="RII", initial_value=int(init["RII"].iloc[-1])
            ),
            BMP2_Alk3 := gillespy2.Species(
                name="BMP2_Alk3",
                initial_value=int(init["BMP2_Alk3"].iloc[-1]),
            ),
            BMP2_Alk8 := gillespy2.Species(
                name="BMP2_Alk8",
                initial_value=int(init["BMP2_Alk8"].iloc[-1]),
            ),
            BMP2_RII := gillespy2.Species(
                name="BMP2_RII",
                initial_value=int(init["BMP2_RII"].iloc[-1]),
            ),
            BMP2_Alk3_Alk3 := gillespy2.Species(
                name="BMP2_Alk3_Alk3",
                initial_value=int(init["BMP2_Alk3_Alk3"].iloc[-1]),
            ),
            BMP2_Alk3_Alk8 := gillespy2.Species(
                name="BMP2_Alk3_Alk8",
                initial_value=int(init["BMP2_Alk3_Alk8"].iloc[-1]),
            ),
            BMP2_Alk3_RII := gillespy2.Species(
                name="BMP2_Alk3_RII",
                initial_value=int(init["BMP2_Alk3_RII"].iloc[-1]),
            ),
            BMP2_Alk8_Alk8 := gillespy2.Species(
                name="BMP2_Alk8_Alk8",
                initial_value=int(init["BMP2_Alk8_Alk8"].iloc[-1]),
            ),
            BMP2_Alk8_RII := gillespy2.Species(
                name="BMP2_Alk8_RII",
                initial_value=int(init["BMP2_Alk8_RII"].iloc[-1]),
            ),
            BMP2_RII_RII := gillespy2.Species(
                name="BMP2_RII_RII",
                initial_value=int(init["BMP2_RII_RII"].iloc[-1]),
            ),
            BMP2_Alk3_Alk3_RII := gillespy2.Species(
                name="BMP2_Alk3_Alk3_RII",
                initial_value=int(init["BMP2_Alk3_Alk3_RII"].iloc[-1]),
            ),
            BMP2_Alk3_Alk8_RII := gillespy2.Species(
                name="BMP2_Alk3_Alk8_RII",
                initial_value=int(init["BMP2_Alk3_Alk8_RII"].iloc[-1]),
            ),
            BMP2_Alk3_RII_RII := gillespy2.Species(
                name="BMP2_Alk3_RII_RII",
                initial_value=int(init["BMP2_Alk3_RII_RII"].iloc[-1]),
            ),
            BMP2_Alk8_Alk8_RII := gillespy2.Species(
                name="BMP2_Alk8_Alk8_RII",
                initial_value=int(init["BMP2_Alk8_Alk8_RII"].iloc[-1]),
            ),
            BMP2_Alk8_RII_RII := gillespy2.Species(
                name="BMP2_Alk8_RII_RII",
                initial_value=int(init["BMP2_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP2_Alk3_Alk3_RII_RII := gillespy2.Species(
                name="BMP2_Alk3_Alk3_RII_RII",
                initial_value=int(init["BMP2_Alk3_Alk3_RII_RII"].iloc[-1]),
            ),
            BMP2_Alk3_Alk8_RII_RII := gillespy2.Species(
                name="BMP2_Alk3_Alk8_RII_RII",
                initial_value=int(init["BMP2_Alk3_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP2_Alk8_Alk8_RII_RII := gillespy2.Species(
                name="BMP2_Alk8_Alk8_RII_RII",
                initial_value=int(init["BMP2_Alk8_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP7_Alk3 := gillespy2.Species(
                name="BMP7_Alk3",
                initial_value=int(init["BMP7_Alk3"].iloc[-1]),
            ),
            BMP7_Alk8 := gillespy2.Species(
                name="BMP7_Alk8",
                initial_value=int(init["BMP7_Alk8"].iloc[-1]),
            ),
            BMP7_RII := gillespy2.Species(
                name="BMP7_RII",
                initial_value=int(init["BMP7_RII"].iloc[-1]),
            ),
            BMP7_Alk3_Alk3 := gillespy2.Species(
                name="BMP7_Alk3_Alk3",
                initial_value=int(init["BMP7_Alk3_Alk3"].iloc[-1]),
            ),
            BMP7_Alk3_Alk8 := gillespy2.Species(
                name="BMP7_Alk3_Alk8",
                initial_value=int(init["BMP7_Alk3_Alk8"].iloc[-1]),
            ),
            BMP7_Alk3_RII := gillespy2.Species(
                name="BMP7_Alk3_RII",
                initial_value=int(init["BMP7_Alk3_RII"].iloc[-1]),
            ),
            BMP7_Alk8_Alk8 := gillespy2.Species(
                name="BMP7_Alk8_Alk8",
                initial_value=int(init["BMP7_Alk8_Alk8"].iloc[-1]),
            ),
            BMP7_Alk8_RII := gillespy2.Species(
                name="BMP7_Alk8_RII",
                initial_value=int(init["BMP7_Alk8_RII"].iloc[-1]),
            ),
            BMP7_RII_RII := gillespy2.Species(
                name="BMP7_RII_RII",
                initial_value=int(init["BMP7_RII_RII"].iloc[-1]),
            ),
            BMP7_Alk3_Alk3_RII := gillespy2.Species(
                name="BMP7_Alk3_Alk3_RII",
                initial_value=int(init["BMP7_Alk3_Alk3_RII"].iloc[-1]),
            ),
            BMP7_Alk3_Alk8_RII := gillespy2.Species(
                name="BMP7_Alk3_Alk8_RII",
                initial_value=int(init["BMP7_Alk3_Alk8_RII"].iloc[-1]),
            ),
            BMP7_Alk3_RII_RII := gillespy2.Species(
                name="BMP7_Alk3_RII_RII",
                initial_value=int(init["BMP7_Alk3_RII_RII"].iloc[-1]),
            ),
            BMP7_Alk8_Alk8_RII := gillespy2.Species(
                name="BMP7_Alk8_Alk8_RII",
                initial_value=int(init["BMP7_Alk8_Alk8_RII"].iloc[-1]),
            ),
            BMP7_Alk8_RII_RII := gillespy2.Species(
                name="BMP7_Alk8_RII_RII",
                initial_value=int(init["BMP7_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP7_Alk3_Alk3_RII_RII := gillespy2.Species(
                name="BMP7_Alk3_Alk3_RII_RII",
                initial_value=int(init["BMP7_Alk3_Alk3_RII_RII"].iloc[-1]),
            ),
            BMP7_Alk3_Alk8_RII_RII := gillespy2.Species(
                name="BMP7_Alk3_Alk8_RII_RII",
                initial_value=int(init["BMP7_Alk3_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP7_Alk8_Alk8_RII_RII := gillespy2.Species(
                name="BMP7_Alk8_Alk8_RII_RII",
                initial_value=int(init["BMP7_Alk8_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP27_Alk3 := gillespy2.Species(
                name="BMP27_Alk3",
                initial_value=int(init["BMP27_Alk3"].iloc[-1]),
            ),
            BMP27_Alk8 := gillespy2.Species(
                name="BMP27_Alk8",
                initial_value=int(init["BMP27_Alk8"].iloc[-1]),
            ),
            BMP27_RII := gillespy2.Species(
                name="BMP27_RII",
                initial_value=int(init["BMP27_RII"].iloc[-1]),
            ),
            BMP27_Alk3_Alk3 := gillespy2.Species(
                name="BMP27_Alk3_Alk3",
                initial_value=int(init["BMP27_Alk3_Alk3"].iloc[-1]),
            ),
            BMP27_Alk3_Alk8 := gillespy2.Species(
                name="BMP27_Alk3_Alk8",
                initial_value=int(init["BMP27_Alk3_Alk8"].iloc[-1]),
            ),
            BMP27_Alk3_RII := gillespy2.Species(
                name="BMP27_Alk3_RII",
                initial_value=int(init["BMP27_Alk3_RII"].iloc[-1]),
            ),
            BMP27_Alk8_Alk8 := gillespy2.Species(
                name="BMP27_Alk8_Alk8",
                initial_value=int(init["BMP27_Alk8_Alk8"].iloc[-1]),
            ),
            BMP27_Alk8_RII := gillespy2.Species(
                name="BMP27_Alk8_RII",
                initial_value=int(init["BMP27_Alk8_RII"].iloc[-1]),
            ),
            BMP27_RII_RII := gillespy2.Species(
                name="BMP27_RII_RII",
                initial_value=int(init["BMP27_RII_RII"].iloc[-1]),
            ),
            BMP27_Alk3_Alk3_RII := gillespy2.Species(
                name="BMP27_Alk3_Alk3_RII",
                initial_value=int(init["BMP27_Alk3_Alk3_RII"].iloc[-1]),
            ),
            BMP27_Alk3_Alk8_RII := gillespy2.Species(
                name="BMP27_Alk3_Alk8_RII",
                initial_value=int(init["BMP27_Alk3_Alk8_RII"].iloc[-1]),
            ),
            BMP27_Alk3_RII_RII := gillespy2.Species(
                name="BMP27_Alk3_RII_RII",
                initial_value=int(init["BMP27_Alk3_RII_RII"].iloc[-1]),
            ),
            BMP27_Alk8_Alk8_RII := gillespy2.Species(
                name="BMP27_Alk8_Alk8_RII",
                initial_value=int(init["BMP27_Alk8_Alk8_RII"].iloc[-1]),
            ),
            BMP27_Alk8_RII_RII := gillespy2.Species(
                name="BMP27_Alk8_RII_RII",
                initial_value=int(init["BMP27_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP27_Alk3_Alk3_RII_RII := gillespy2.Species(
                name="BMP27_Alk3_Alk3_RII_RII",
                initial_value=int(init["BMP27_Alk3_Alk3_RII_RII"].iloc[-1]),
            ),
            BMP27_Alk3_Alk8_RII_RII := gillespy2.Species(
                name="BMP27_Alk3_Alk8_RII_RII",
                initial_value=int(init["BMP27_Alk3_Alk8_RII_RII"].iloc[-1]),
            ),
            BMP27_Alk8_Alk8_RII_RII := gillespy2.Species(
                name="BMP27_Alk8_Alk8_RII_RII",
                initial_value=int(init["BMP27_Alk8_Alk8_RII_RII"].iloc[-1]),
            ),
        ]
    )

    # Reactions
    model.add_reaction(
        [
            rxn1 := gillespy2.Reaction(
                name="BMP2_Alk3 production 1",
                reactants={Alk3: 1},
                products={BMP2_Alk3: 1},
                rate=k1A,
            ),
            rxn2 := gillespy2.Reaction(
                name="BMP2_Alk3 dissolution 2",
                reactants={BMP2_Alk3: 1},
                products={Alk3: 1},
                rate=k1r,
            ),
            rxn3 := gillespy2.Reaction(
                name="BMP2_Alk8 production 3",
                reactants={Alk8: 1},
                products={BMP2_Alk8: 1},
                rate=k2A,
            ),
            rxn4 := gillespy2.Reaction(
                name="BMP2_Alk8 dissolution 4",
                reactants={BMP2_Alk8: 1},
                products={Alk8: 1},
                rate=k2r,
            ),
            rxn5 := gillespy2.Reaction(
                name="BMP2_RII production 5",
                reactants={RII: 1},
                products={BMP2_RII: 1},
                rate=k3A,
            ),
            rxn6 := gillespy2.Reaction(
                name="BMP2_RII dissolution 6",
                reactants={BMP2_RII: 1},
                products={RII: 1},
                rate=k3r,
            ),
            rxn7 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3 production 7",
                reactants={BMP2_Alk3: 1, Alk3: 1},
                products={BMP2_Alk3_Alk3: 1},
                rate=k4,
            ),
            rxn8 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3 dissolution 8",
                reactants={BMP2_Alk3_Alk3: 1},
                products={BMP2_Alk3: 1, Alk3: 1},
                rate=k4r,
            ),
            rxn9 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8 production 9",
                reactants={BMP2_Alk3: 1, Alk8: 1},
                products={BMP2_Alk3_Alk8: 1},
                rate=k5,
            ),
            rxn10 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8 dissolution 10",
                reactants={BMP2_Alk3_Alk8: 1},
                products={BMP2_Alk3: 1, Alk8: 1},
                rate=k5r,
            ),
            rxn11 := gillespy2.Reaction(
                name="BMP2_Alk3_RII production 11",
                reactants={BMP2_Alk3: 1, RII: 1},
                products={BMP2_Alk3_RII: 1},
                rate=k6,
            ),
            rxn12 := gillespy2.Reaction(
                name="BMP2_Alk3_RII dissolution 12",
                reactants={BMP2_Alk3_RII: 1},
                products={BMP2_Alk3: 1, RII: 1},
                rate=k6r,
            ),
            rxn13 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8 production 13",
                reactants={BMP2_Alk8: 1, Alk3: 1},
                products={BMP2_Alk3_Alk8: 1},
                rate=k7,
            ),
            rxn14 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8 dissolution 14",
                reactants={BMP2_Alk3_Alk8: 1},
                products={BMP2_Alk8: 1, Alk3: 1},
                rate=k7r,
            ),
            rxn15 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8 production 15",
                reactants={BMP2_Alk8: 1, Alk8: 1},
                products={BMP2_Alk8_Alk8: 1},
                rate=k8,
            ),
            rxn16 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8 dissolution 16",
                reactants={BMP2_Alk8_Alk8: 1},
                products={BMP2_Alk8: 1, Alk8: 1},
                rate=k8r,
            ),
            rxn17 := gillespy2.Reaction(
                name="BMP2_Alk8_RII production 17",
                reactants={BMP2_Alk8: 1, RII: 1},
                products={BMP2_Alk8_RII: 1},
                rate=k9,
            ),
            rxn18 := gillespy2.Reaction(
                name="BMP2_Alk8_RII dissolution 18",
                reactants={BMP2_Alk8_RII: 1},
                products={BMP2_Alk8: 1, RII: 1},
                rate=k9r,
            ),
            rxn19 := gillespy2.Reaction(
                name="BMP2_Alk3_RII production 19",
                reactants={BMP2_RII: 1, Alk3: 1},
                products={BMP2_Alk3_RII: 1},
                rate=k10,
            ),
            rxn20 := gillespy2.Reaction(
                name="BMP2_Alk3_RII dissolution 20",
                reactants={BMP2_Alk3_RII: 1},
                products={BMP2_RII: 1, Alk3: 1},
                rate=k10r,
            ),
            rxn21 := gillespy2.Reaction(
                name="BMP2_Alk8_RII production 21",
                reactants={BMP2_RII: 1, Alk8: 1},
                products={BMP2_Alk8_RII: 1},
                rate=k11,
            ),
            rxn22 := gillespy2.Reaction(
                name="BMP2_Alk8_RII dissolution 22",
                reactants={BMP2_Alk8_RII: 1},
                products={BMP2_RII: 1, Alk8: 1},
                rate=k11r,
            ),
            rxn23 := gillespy2.Reaction(
                name="BMP2_RII_RII production 23",
                reactants={BMP2_RII: 1, RII: 1},
                products={BMP2_RII_RII: 1},
                rate=k12,
            ),
            rxn24 := gillespy2.Reaction(
                name="BMP2_RII_RII dissolution 24",
                reactants={BMP2_RII_RII: 1},
                products={BMP2_RII: 1, RII: 1},
                rate=k12r,
            ),
            rxn25 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII production 25",
                reactants={BMP2_Alk8_Alk8: 1, RII: 1},
                products={BMP2_Alk8_Alk8_RII: 1},
                rate=k13,
            ),
            rxn26 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII dissolution 26",
                reactants={BMP2_Alk8_Alk8_RII: 1},
                products={BMP2_Alk8_Alk8: 1, RII: 1},
                rate=k13r,
            ),
            rxn27 := gillespy2.Reaction(
                name="BMP2_Alk3_RII_RII production 27",
                reactants={BMP2_RII_RII: 1, Alk3: 1},
                products={BMP2_Alk3_RII_RII: 1},
                rate=k14,
            ),
            rxn28 := gillespy2.Reaction(
                name="BMP2_Alk3_RII_RII dissolution 28",
                reactants={BMP2_Alk3_RII_RII: 1},
                products={BMP2_RII_RII: 1, Alk3: 1},
                rate=k14r,
            ),
            rxn29 := gillespy2.Reaction(
                name="BMP2_Alk8_RII_RII production 29",
                reactants={BMP2_RII_RII: 1, Alk8: 1},
                products={BMP2_Alk8_RII_RII: 1},
                rate=k15,
            ),
            rxn30 := gillespy2.Reaction(
                name="BMP2_Alk8_RII_RII dissolution 30",
                reactants={BMP2_Alk8_RII_RII: 1},
                products={BMP2_RII_RII: 1, Alk8: 1},
                rate=k15r,
            ),
            rxn31 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII production 31",
                reactants={BMP2_Alk3_Alk3: 1, RII: 1},
                products={BMP2_Alk3_Alk3_RII: 1},
                rate=k16,
            ),
            rxn32 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII dissolution 32",
                reactants={BMP2_Alk3_Alk3_RII: 1},
                products={BMP2_Alk3_Alk3: 1, RII: 1},
                rate=k16r,
            ),
            rxn33 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII production 33",
                reactants={BMP2_Alk3_Alk8: 1, RII: 1},
                products={BMP2_Alk3_Alk8_RII: 1},
                rate=k17,
            ),
            rxn34 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII dissolution 34",
                reactants={BMP2_Alk3_Alk8_RII: 1},
                products={BMP2_Alk3_Alk8: 1, RII: 1},
                rate=k17r,
            ),
            rxn35 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII production 35",
                reactants={BMP2_Alk3_RII: 1, Alk3: 1},
                products={BMP2_Alk3_Alk3_RII: 1},
                rate=k18,
            ),
            rxn36 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII dissolution 36",
                reactants={BMP2_Alk3_Alk3_RII: 1},
                products={BMP2_Alk3_RII: 1, Alk3: 1},
                rate=k18r,
            ),
            rxn37 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII production 37",
                reactants={BMP2_Alk3_RII: 1, Alk8: 1},
                products={BMP2_Alk3_Alk8_RII: 1},
                rate=k19,
            ),
            rxn38 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII dissolution 38",
                reactants={BMP2_Alk3_Alk8_RII: 1},
                products={BMP2_Alk3_RII: 1, Alk8: 1},
                rate=k19r,
            ),
            rxn39 := gillespy2.Reaction(
                name="BMP2_Alk3_RII_RII production 39",
                reactants={BMP2_Alk3_RII: 1, RII: 1},
                products={BMP2_Alk3_RII_RII: 1},
                rate=k20,
            ),
            rxn40 := gillespy2.Reaction(
                name="BMP2_Alk3_RII_RII dissolution 40",
                reactants={BMP2_Alk3_RII_RII: 1},
                products={BMP2_Alk3_RII: 1, RII: 1},
                rate=k20r,
            ),
            rxn41 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII production 41",
                reactants={BMP2_Alk8_RII: 1, Alk3: 1},
                products={BMP2_Alk3_Alk8_RII: 1},
                rate=k21,
            ),
            rxn42 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII dissolution 42",
                reactants={BMP2_Alk3_Alk8_RII: 1},
                products={BMP2_Alk8_RII: 1, Alk3: 1},
                rate=k21r,
            ),
            rxn43 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII production 43",
                reactants={BMP2_Alk8_RII: 1, Alk8: 1},
                products={BMP2_Alk8_Alk8_RII: 1},
                rate=k22,
            ),
            rxn44 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII dissolution 44",
                reactants={BMP2_Alk8_Alk8_RII: 1},
                products={BMP2_Alk8_RII: 1, Alk8: 1},
                rate=k22r,
            ),
            rxn45 := gillespy2.Reaction(
                name="BMP2_Alk8_RII_RII production 45",
                reactants={BMP2_Alk8_RII: 1, RII: 1},
                products={BMP2_Alk8_RII_RII: 1},
                rate=k23,
            ),
            rxn46 := gillespy2.Reaction(
                name="BMP2_Alk8_RII_RII dissolution 46",
                reactants={BMP2_Alk8_RII_RII: 1},
                products={BMP2_Alk8_RII: 1, RII: 1},
                rate=k23r,
            ),
            rxn47 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII_RII production 47",
                reactants={BMP2_Alk3_Alk3_RII: 1, RII: 1},
                products={BMP2_Alk3_Alk3_RII_RII: 1},
                rate=k24,
            ),
            rxn48 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII_RII dissolution 48",
                reactants={BMP2_Alk3_Alk3_RII_RII: 1},
                products={BMP2_Alk3_Alk3_RII: 1, RII: 1},
                rate=k24r,
            ),
            rxn49 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII_RII production 49",
                reactants={BMP2_Alk3_Alk8_RII: 1, RII: 1},
                products={BMP2_Alk3_Alk8_RII_RII: 1},
                rate=k25,
            ),
            rxn50 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII_RII dissolution 50",
                reactants={BMP2_Alk3_Alk8_RII_RII: 1},
                products={BMP2_Alk3_Alk8_RII: 1, RII: 1},
                rate=k25r,
            ),
            rxn51 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII_RII production 51",
                reactants={BMP2_Alk3_RII_RII: 1, Alk3: 1},
                products={BMP2_Alk3_Alk3_RII_RII: 1},
                rate=k26,
            ),
            rxn52 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk3_RII_RII dissolution 52",
                reactants={BMP2_Alk3_Alk3_RII_RII: 1},
                products={BMP2_Alk3_RII_RII: 1, Alk3: 1},
                rate=k26r,
            ),
            rxn53 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII_RII production 53",
                reactants={BMP2_Alk3_RII_RII: 1, Alk8: 1},
                products={BMP2_Alk3_Alk8_RII_RII: 1},
                rate=k27,
            ),
            rxn54 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII_RII dissolution 54",
                reactants={BMP2_Alk3_Alk8_RII_RII: 1},
                products={BMP2_Alk3_RII_RII: 1, Alk8: 1},
                rate=k27r,
            ),
            rxn55 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII_RII production 55",
                reactants={BMP2_Alk8_Alk8_RII: 1, RII: 1},
                products={BMP2_Alk8_Alk8_RII_RII: 1},
                rate=k28,
            ),
            rxn56 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII_RII dissolution 56",
                reactants={BMP2_Alk8_Alk8_RII_RII: 1},
                products={BMP2_Alk8_Alk8_RII: 1, RII: 1},
                rate=k28r,
            ),
            rxn57 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII_RII production 57",
                reactants={BMP2_Alk8_RII_RII: 1, Alk3: 1},
                products={BMP2_Alk3_Alk8_RII_RII: 1},
                rate=k29,
            ),
            rxn58 := gillespy2.Reaction(
                name="BMP2_Alk3_Alk8_RII_RII dissolution 58",
                reactants={BMP2_Alk3_Alk8_RII_RII: 1},
                products={BMP2_Alk8_RII_RII: 1, Alk3: 1},
                rate=k29r,
            ),
            rxn59 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII_RII production 59",
                reactants={BMP2_Alk8_RII_RII: 1, Alk8: 1},
                products={BMP2_Alk8_Alk8_RII_RII: 1},
                rate=k30,
            ),
            rxn60 := gillespy2.Reaction(
                name="BMP2_Alk8_Alk8_RII_RII dissolution 60",
                reactants={BMP2_Alk8_Alk8_RII_RII: 1},
                products={BMP2_Alk8_RII_RII: 1, Alk8: 1},
                rate=k30r,
            ),
            rxn61 := gillespy2.Reaction(
                name="BMP7_Alk3 production 1",
                reactants={Alk3: 1},
                products={BMP7_Alk3: 1},
                rate=k31A,
            ),
            rxn62 := gillespy2.Reaction(
                name="BMP7_Alk3 dissolution 2",
                reactants={BMP7_Alk3: 1},
                products={Alk3: 1},
                rate=k31r,
            ),
            rxn63 := gillespy2.Reaction(
                name="BMP7_Alk8 production 3",
                reactants={Alk8: 1},
                products={BMP7_Alk8: 1},
                rate=k32A,
            ),
            rxn64 := gillespy2.Reaction(
                name="BMP7_Alk8 dissolution 4",
                reactants={BMP7_Alk8: 1},
                products={Alk8: 1},
                rate=k32r,
            ),
            rxn65 := gillespy2.Reaction(
                name="BMP7_RII production 5",
                reactants={RII: 1},
                products={BMP7_RII: 1},
                rate=k33A,
            ),
            rxn66 := gillespy2.Reaction(
                name="BMP7_RII dissolution 6",
                reactants={BMP7_RII: 1},
                products={RII: 1},
                rate=k33r,
            ),
            rxn67 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3 production 7",
                reactants={BMP7_Alk3: 1, Alk3: 1},
                products={BMP7_Alk3_Alk3: 1},
                rate=k34,
            ),
            rxn68 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3 dissolution 8",
                reactants={BMP7_Alk3_Alk3: 1},
                products={BMP7_Alk3: 1, Alk3: 1},
                rate=k34r,
            ),
            rxn69 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8 production 9",
                reactants={BMP7_Alk3: 1, Alk8: 1},
                products={BMP7_Alk3_Alk8: 1},
                rate=k35,
            ),
            rxn70 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8 dissolution 10",
                reactants={BMP7_Alk3_Alk8: 1},
                products={BMP7_Alk3: 1, Alk8: 1},
                rate=k35r,
            ),
            rxn71 := gillespy2.Reaction(
                name="BMP7_Alk3_RII production 11",
                reactants={BMP7_Alk3: 1, RII: 1},
                products={BMP7_Alk3_RII: 1},
                rate=k36,
            ),
            rxn72 := gillespy2.Reaction(
                name="BMP7_Alk3_RII dissolution 12",
                reactants={BMP7_Alk3_RII: 1},
                products={BMP7_Alk3: 1, RII: 1},
                rate=k36r,
            ),
            rxn73 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8 production 13",
                reactants={BMP7_Alk8: 1, Alk3: 1},
                products={BMP7_Alk3_Alk8: 1},
                rate=k37,
            ),
            rxn74 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8 dissolution 14",
                reactants={BMP7_Alk3_Alk8: 1},
                products={BMP7_Alk8: 1, Alk3: 1},
                rate=k37r,
            ),
            rxn75 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8 production 15",
                reactants={BMP7_Alk8: 1, Alk8: 1},
                products={BMP7_Alk8_Alk8: 1},
                rate=k38,
            ),
            rxn76 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8 dissolution 16",
                reactants={BMP7_Alk8_Alk8: 1},
                products={BMP7_Alk8: 1, Alk8: 1},
                rate=k38r,
            ),
            rxn77 := gillespy2.Reaction(
                name="BMP7_Alk8_RII production 17",
                reactants={BMP7_Alk8: 1, RII: 1},
                products={BMP7_Alk8_RII: 1},
                rate=k39,
            ),
            rxn78 := gillespy2.Reaction(
                name="BMP7_Alk8_RII dissolution 18",
                reactants={BMP7_Alk8_RII: 1},
                products={BMP7_Alk8: 1, RII: 1},
                rate=k39r,
            ),
            rxn79 := gillespy2.Reaction(
                name="BMP7_Alk3_RII production 19",
                reactants={BMP7_RII: 1, Alk3: 1},
                products={BMP7_Alk3_RII: 1},
                rate=k40,
            ),
            rxn80 := gillespy2.Reaction(
                name="BMP7_Alk3_RII dissolution 20",
                reactants={BMP7_Alk3_RII: 1},
                products={BMP7_RII: 1, Alk3: 1},
                rate=k40r,
            ),
            rxn81 := gillespy2.Reaction(
                name="BMP7_Alk8_RII production 21",
                reactants={BMP7_RII: 1, Alk8: 1},
                products={BMP7_Alk8_RII: 1},
                rate=k41,
            ),
            rxn82 := gillespy2.Reaction(
                name="BMP7_Alk8_RII dissolution 22",
                reactants={BMP7_Alk8_RII: 1},
                products={BMP7_RII: 1, Alk8: 1},
                rate=k41r,
            ),
            rxn83 := gillespy2.Reaction(
                name="BMP7_RII_RII production 23",
                reactants={BMP7_RII: 1, RII: 1},
                products={BMP7_RII_RII: 1},
                rate=k42,
            ),
            rxn84 := gillespy2.Reaction(
                name="BMP7_RII_RII dissolution 24",
                reactants={BMP7_RII_RII: 1},
                products={BMP7_RII: 1, RII: 1},
                rate=k42r,
            ),
            rxn85 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII production 25",
                reactants={BMP7_Alk8_Alk8: 1, RII: 1},
                products={BMP7_Alk8_Alk8_RII: 1},
                rate=k43,
            ),
            rxn86 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII dissolution 26",
                reactants={BMP7_Alk8_Alk8_RII: 1},
                products={BMP7_Alk8_Alk8: 1, RII: 1},
                rate=k43r,
            ),
            rxn87 := gillespy2.Reaction(
                name="BMP7_Alk3_RII_RII production 27",
                reactants={BMP7_RII_RII: 1, Alk3: 1},
                products={BMP7_Alk3_RII_RII: 1},
                rate=k44,
            ),
            rxn88 := gillespy2.Reaction(
                name="BMP7_Alk3_RII_RII dissolution 28",
                reactants={BMP7_Alk3_RII_RII: 1},
                products={BMP7_RII_RII: 1, Alk3: 1},
                rate=k44r,
            ),
            rxn89 := gillespy2.Reaction(
                name="BMP7_Alk8_RII_RII production 29",
                reactants={BMP7_RII_RII: 1, Alk8: 1},
                products={BMP7_Alk8_RII_RII: 1},
                rate=k45,
            ),
            rxn90 := gillespy2.Reaction(
                name="BMP7_Alk8_RII_RII dissolution 30",
                reactants={BMP7_Alk8_RII_RII: 1},
                products={BMP7_RII_RII: 1, Alk8: 1},
                rate=k45r,
            ),
            rxn91 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII production 31",
                reactants={BMP7_Alk3_Alk3: 1, RII: 1},
                products={BMP7_Alk3_Alk3_RII: 1},
                rate=k46,
            ),
            rxn92 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII dissolution 32",
                reactants={BMP7_Alk3_Alk3_RII: 1},
                products={BMP7_Alk3_Alk3: 1, RII: 1},
                rate=k46r,
            ),
            rxn93 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII production 33",
                reactants={BMP7_Alk3_Alk8: 1, RII: 1},
                products={BMP7_Alk3_Alk8_RII: 1},
                rate=k47,
            ),
            rxn94 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII dissolution 34",
                reactants={BMP7_Alk3_Alk8_RII: 1},
                products={BMP7_Alk3_Alk8: 1, RII: 1},
                rate=k47r,
            ),
            rxn95 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII production 35",
                reactants={BMP7_Alk3_RII: 1, Alk3: 1},
                products={BMP7_Alk3_Alk3_RII: 1},
                rate=k48,
            ),
            rxn96 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII dissolution 36",
                reactants={BMP7_Alk3_Alk3_RII: 1},
                products={BMP7_Alk3_RII: 1, Alk3: 1},
                rate=k48r,
            ),
            rxn97 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII production 37",
                reactants={BMP7_Alk3_RII: 1, Alk8: 1},
                products={BMP7_Alk3_Alk8_RII: 1},
                rate=k49,
            ),
            rxn98 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII dissolution 38",
                reactants={BMP7_Alk3_Alk8_RII: 1},
                products={BMP7_Alk3_RII: 1, Alk8: 1},
                rate=k49r,
            ),
            rxn99 := gillespy2.Reaction(
                name="BMP7_Alk3_RII_RII production 39",
                reactants={BMP7_Alk3_RII: 1, RII: 1},
                products={BMP7_Alk3_RII_RII: 1},
                rate=k50,
            ),
            rxn100 := gillespy2.Reaction(
                name="BMP7_Alk3_RII_RII dissolution 40",
                reactants={BMP7_Alk3_RII_RII: 1},
                products={BMP7_Alk3_RII: 1, RII: 1},
                rate=k50r,
            ),
            rxn101 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII production 41",
                reactants={BMP7_Alk8_RII: 1, Alk3: 1},
                products={BMP7_Alk3_Alk8_RII: 1},
                rate=k51,
            ),
            rxn102 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII dissolution 42",
                reactants={BMP7_Alk3_Alk8_RII: 1},
                products={BMP7_Alk8_RII: 1, Alk3: 1},
                rate=k51r,
            ),
            rxn103 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII production 43",
                reactants={BMP7_Alk8_RII: 1, Alk8: 1},
                products={BMP7_Alk8_Alk8_RII: 1},
                rate=k52,
            ),
            rxn104 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII dissolution 44",
                reactants={BMP7_Alk8_Alk8_RII: 1},
                products={BMP7_Alk8_RII: 1, Alk8: 1},
                rate=k52r,
            ),
            rxn105 := gillespy2.Reaction(
                name="BMP7_Alk8_RII_RII production 45",
                reactants={BMP7_Alk8_RII: 1, RII: 1},
                products={BMP7_Alk8_RII_RII: 1},
                rate=k53,
            ),
            rxn106 := gillespy2.Reaction(
                name="BMP7_Alk8_RII_RII dissolution 46",
                reactants={BMP7_Alk8_RII_RII: 1},
                products={BMP7_Alk8_RII: 1, RII: 1},
                rate=k53r,
            ),
            rxn107 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII_RII production 47",
                reactants={BMP7_Alk3_Alk3_RII: 1, RII: 1},
                products={BMP7_Alk3_Alk3_RII_RII: 1},
                rate=k54,
            ),
            rxn108 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII_RII dissolution 48",
                reactants={BMP7_Alk3_Alk3_RII_RII: 1},
                products={BMP7_Alk3_Alk3_RII: 1, RII: 1},
                rate=k54r,
            ),
            rxn109 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII_RII production 49",
                reactants={BMP7_Alk3_Alk8_RII: 1, RII: 1},
                products={BMP7_Alk3_Alk8_RII_RII: 1},
                rate=k55,
            ),
            rxn110 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII_RII dissolution 50",
                reactants={BMP7_Alk3_Alk8_RII_RII: 1},
                products={BMP7_Alk3_Alk8_RII: 1, RII: 1},
                rate=k55r,
            ),
            rxn111 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII_RII production 51",
                reactants={BMP7_Alk3_RII_RII: 1, Alk3: 1},
                products={BMP7_Alk3_Alk3_RII_RII: 1},
                rate=k56,
            ),
            rxn112 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk3_RII_RII dissolution 52",
                reactants={BMP7_Alk3_Alk3_RII_RII: 1},
                products={BMP7_Alk3_RII_RII: 1, Alk3: 1},
                rate=k56r,
            ),
            rxn113 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII_RII production 53",
                reactants={BMP7_Alk3_RII_RII: 1, Alk8: 1},
                products={BMP7_Alk3_Alk8_RII_RII: 1},
                rate=k57,
            ),
            rxn114 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII_RII dissolution 54",
                reactants={BMP7_Alk3_Alk8_RII_RII: 1},
                products={BMP7_Alk3_RII_RII: 1, Alk8: 1},
                rate=k57r,
            ),
            rxn115 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII_RII production 55",
                reactants={BMP7_Alk8_Alk8_RII: 1, RII: 1},
                products={BMP7_Alk8_Alk8_RII_RII: 1},
                rate=k58,
            ),
            rxn116 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII_RII dissolution 56",
                reactants={BMP7_Alk8_Alk8_RII_RII: 1},
                products={BMP7_Alk8_Alk8_RII: 1, RII: 1},
                rate=k58r,
            ),
            rxn117 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII_RII production 57",
                reactants={BMP7_Alk8_RII_RII: 1, Alk3: 1},
                products={BMP7_Alk3_Alk8_RII_RII: 1},
                rate=k59,
            ),
            rxn118 := gillespy2.Reaction(
                name="BMP7_Alk3_Alk8_RII_RII dissolution 58",
                reactants={BMP7_Alk3_Alk8_RII_RII: 1},
                products={BMP7_Alk8_RII_RII: 1, Alk3: 1},
                rate=k59r,
            ),
            rxn119 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII_RII production 59",
                reactants={BMP7_Alk8_RII_RII: 1, Alk8: 1},
                products={BMP7_Alk8_Alk8_RII_RII: 1},
                rate=k60,
            ),
            rxn120 := gillespy2.Reaction(
                name="BMP7_Alk8_Alk8_RII_RII dissolution 60",
                reactants={BMP7_Alk8_Alk8_RII_RII: 1},
                products={BMP7_Alk8_RII_RII: 1, Alk8: 1},
                rate=k60r,
            ),
            rxn121 := gillespy2.Reaction(
                name="BMP27_Alk3 production 1",
                reactants={Alk3: 1},
                products={BMP27_Alk3: 1},
                rate=k61A,
            ),
            rxn122 := gillespy2.Reaction(
                name="BMP27_Alk3 dissolution 2",
                reactants={BMP27_Alk3: 1},
                products={Alk3: 1},
                rate=k61r,
            ),
            rxn123 := gillespy2.Reaction(
                name="BMP27_Alk8 production 3",
                reactants={Alk8: 1},
                products={BMP27_Alk8: 1},
                rate=k62A,
            ),
            rxn124 := gillespy2.Reaction(
                name="BMP27_Alk8 dissolution 4",
                reactants={BMP27_Alk8: 1},
                products={Alk8: 1},
                rate=k62r,
            ),
            rxn125 := gillespy2.Reaction(
                name="BMP27_RII production 5",
                reactants={RII: 1},
                products={BMP27_RII: 1},
                rate=k63A,
            ),
            rxn126 := gillespy2.Reaction(
                name="BMP27_RII dissolution 6",
                reactants={BMP27_RII: 1},
                products={RII: 1},
                rate=k63r,
            ),
            rxn127 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3 production 7",
                reactants={BMP27_Alk3: 1, Alk3: 1},
                products={BMP27_Alk3_Alk3: 1},
                rate=k64,
            ),
            rxn128 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3 dissolution 8",
                reactants={BMP27_Alk3_Alk3: 1},
                products={BMP27_Alk3: 1, Alk3: 1},
                rate=k64r,
            ),
            rxn129 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8 production 9",
                reactants={BMP27_Alk3: 1, Alk8: 1},
                products={BMP27_Alk3_Alk8: 1},
                rate=k65,
            ),
            rxn130 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8 dissolution 10",
                reactants={BMP27_Alk3_Alk8: 1},
                products={BMP27_Alk3: 1, Alk8: 1},
                rate=k65r,
            ),
            rxn131 := gillespy2.Reaction(
                name="BMP27_Alk3_RII production 11",
                reactants={BMP27_Alk3: 1, RII: 1},
                products={BMP27_Alk3_RII: 1},
                rate=k66,
            ),
            rxn132 := gillespy2.Reaction(
                name="BMP27_Alk3_RII dissolution 12",
                reactants={BMP27_Alk3_RII: 1},
                products={BMP27_Alk3: 1, RII: 1},
                rate=k66r,
            ),
            rxn133 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8 production 13",
                reactants={BMP27_Alk8: 1, Alk3: 1},
                products={BMP27_Alk3_Alk8: 1},
                rate=k67,
            ),
            rxn134 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8 dissolution 14",
                reactants={BMP27_Alk3_Alk8: 1},
                products={BMP27_Alk8: 1, Alk3: 1},
                rate=k67r,
            ),
            rxn135 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8 production 15",
                reactants={BMP27_Alk8: 1, Alk8: 1},
                products={BMP27_Alk8_Alk8: 1},
                rate=k68,
            ),
            rxn136 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8 dissolution 16",
                reactants={BMP27_Alk8_Alk8: 1},
                products={BMP27_Alk8: 1, Alk8: 1},
                rate=k68r,
            ),
            rxn137 := gillespy2.Reaction(
                name="BMP27_Alk8_RII production 17",
                reactants={BMP27_Alk8: 1, RII: 1},
                products={BMP27_Alk8_RII: 1},
                rate=k69,
            ),
            rxn138 := gillespy2.Reaction(
                name="BMP27_Alk8_RII dissolution 18",
                reactants={BMP27_Alk8_RII: 1},
                products={BMP27_Alk8: 1, RII: 1},
                rate=k69r,
            ),
            rxn139 := gillespy2.Reaction(
                name="BMP27_Alk3_RII production 19",
                reactants={BMP27_RII: 1, Alk3: 1},
                products={BMP27_Alk3_RII: 1},
                rate=k70,
            ),
            rxn140 := gillespy2.Reaction(
                name="BMP27_Alk3_RII dissolution 20",
                reactants={BMP27_Alk3_RII: 1},
                products={BMP27_RII: 1, Alk3: 1},
                rate=k70r,
            ),
            rxn141 := gillespy2.Reaction(
                name="BMP27_Alk8_RII production 21",
                reactants={BMP27_RII: 1, Alk8: 1},
                products={BMP27_Alk8_RII: 1},
                rate=k71,
            ),
            rxn142 := gillespy2.Reaction(
                name="BMP27_Alk8_RII dissolution 22",
                reactants={BMP27_Alk8_RII: 1},
                products={BMP27_RII: 1, Alk8: 1},
                rate=k71r,
            ),
            rxn143 := gillespy2.Reaction(
                name="BMP27_RII_RII production 23",
                reactants={BMP27_RII: 1, RII: 1},
                products={BMP27_RII_RII: 1},
                rate=k72,
            ),
            rxn144 := gillespy2.Reaction(
                name="BMP27_RII_RII dissolution 24",
                reactants={BMP27_RII_RII: 1},
                products={BMP27_RII: 1, RII: 1},
                rate=k72r,
            ),
            rxn145 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII production 25",
                reactants={BMP27_Alk8_Alk8: 1, RII: 1},
                products={BMP27_Alk8_Alk8_RII: 1},
                rate=k73,
            ),
            rxn146 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII dissolution 26",
                reactants={BMP27_Alk8_Alk8_RII: 1},
                products={BMP27_Alk8_Alk8: 1, RII: 1},
                rate=k73r,
            ),
            rxn147 := gillespy2.Reaction(
                name="BMP27_Alk3_RII_RII production 27",
                reactants={BMP27_RII_RII: 1, Alk3: 1},
                products={BMP27_Alk3_RII_RII: 1},
                rate=k74,
            ),
            rxn148 := gillespy2.Reaction(
                name="BMP27_Alk3_RII_RII dissolution 28",
                reactants={BMP27_Alk3_RII_RII: 1},
                products={BMP27_RII_RII: 1, Alk3: 1},
                rate=k74r,
            ),
            rxn149 := gillespy2.Reaction(
                name="BMP27_Alk8_RII_RII production 29",
                reactants={BMP27_RII_RII: 1, Alk8: 1},
                products={BMP27_Alk8_RII_RII: 1},
                rate=k75,
            ),
            rxn150 := gillespy2.Reaction(
                name="BMP27_Alk8_RII_RII dissolution 30",
                reactants={BMP27_Alk8_RII_RII: 1},
                products={BMP27_RII_RII: 1, Alk8: 1},
                rate=k75r,
            ),
            rxn151 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII production 31",
                reactants={BMP27_Alk3_Alk3: 1, RII: 1},
                products={BMP27_Alk3_Alk3_RII: 1},
                rate=k76,
            ),
            rxn152 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII dissolution 32",
                reactants={BMP27_Alk3_Alk3_RII: 1},
                products={BMP27_Alk3_Alk3: 1, RII: 1},
                rate=k76r,
            ),
            rxn153 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII production 33",
                reactants={BMP27_Alk3_Alk8: 1, RII: 1},
                products={BMP27_Alk3_Alk8_RII: 1},
                rate=k77,
            ),
            rxn154 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII dissolution 34",
                reactants={BMP27_Alk3_Alk8_RII: 1},
                products={BMP27_Alk3_Alk8: 1, RII: 1},
                rate=k77r,
            ),
            rxn155 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII production 35",
                reactants={BMP27_Alk3_RII: 1, Alk3: 1},
                products={BMP27_Alk3_Alk3_RII: 1},
                rate=k78,
            ),
            rxn156 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII dissolution 36",
                reactants={BMP27_Alk3_Alk3_RII: 1},
                products={BMP27_Alk3_RII: 1, Alk3: 1},
                rate=k78r,
            ),
            rxn157 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII production 37",
                reactants={BMP27_Alk3_RII: 1, Alk8: 1},
                products={BMP27_Alk3_Alk8_RII: 1},
                rate=k79,
            ),
            rxn158 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII dissolution 38",
                reactants={BMP27_Alk3_Alk8_RII: 1},
                products={BMP27_Alk3_RII: 1, Alk8: 1},
                rate=k79r,
            ),
            rxn159 := gillespy2.Reaction(
                name="BMP27_Alk3_RII_RII production 39",
                reactants={BMP27_Alk3_RII: 1, RII: 1},
                products={BMP27_Alk3_RII_RII: 1},
                rate=k80,
            ),
            rxn160 := gillespy2.Reaction(
                name="BMP27_Alk3_RII_RII dissolution 40",
                reactants={BMP27_Alk3_RII_RII: 1},
                products={BMP27_Alk3_RII: 1, RII: 1},
                rate=k80r,
            ),
            rxn161 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII production 41",
                reactants={BMP27_Alk8_RII: 1, Alk3: 1},
                products={BMP27_Alk3_Alk8_RII: 1},
                rate=k81,
            ),
            rxn162 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII dissolution 42",
                reactants={BMP27_Alk3_Alk8_RII: 1},
                products={BMP27_Alk8_RII: 1, Alk3: 1},
                rate=k81r,
            ),
            rxn163 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII production 43",
                reactants={BMP27_Alk8_RII: 1, Alk8: 1},
                products={BMP27_Alk8_Alk8_RII: 1},
                rate=k82,
            ),
            rxn164 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII dissolution 44",
                reactants={BMP27_Alk8_Alk8_RII: 1},
                products={BMP27_Alk8_RII: 1, Alk8: 1},
                rate=k82r,
            ),
            rxn165 := gillespy2.Reaction(
                name="BMP27_Alk8_RII_RII production 45",
                reactants={BMP27_Alk8_RII: 1, RII: 1},
                products={BMP27_Alk8_RII_RII: 1},
                rate=k83,
            ),
            rxn166 := gillespy2.Reaction(
                name="BMP27_Alk8_RII_RII dissolution 46",
                reactants={BMP27_Alk8_RII_RII: 1},
                products={BMP27_Alk8_RII: 1, RII: 1},
                rate=k83r,
            ),
            rxn167 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII_RII production 47",
                reactants={BMP27_Alk3_Alk3_RII: 1, RII: 1},
                products={BMP27_Alk3_Alk3_RII_RII: 1},
                rate=k84,
            ),
            rxn168 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII_RII dissolution 48",
                reactants={BMP27_Alk3_Alk3_RII_RII: 1},
                products={BMP27_Alk3_Alk3_RII: 1, RII: 1},
                rate=k84r,
            ),
            rxn169 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII_RII production 49",
                reactants={BMP27_Alk3_Alk8_RII: 1, RII: 1},
                products={BMP27_Alk3_Alk8_RII_RII: 1},
                rate=k85,
            ),
            rxn170 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII_RII dissolution 50",
                reactants={BMP27_Alk3_Alk8_RII_RII: 1},
                products={BMP27_Alk3_Alk8_RII: 1, RII: 1},
                rate=k85r,
            ),
            rxn171 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII_RII production 51",
                reactants={BMP27_Alk3_RII_RII: 1, Alk3: 1},
                products={BMP27_Alk3_Alk3_RII_RII: 1},
                rate=k86,
            ),
            rxn172 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk3_RII_RII dissolution 52",
                reactants={BMP27_Alk3_Alk3_RII_RII: 1},
                products={BMP27_Alk3_RII_RII: 1, Alk3: 1},
                rate=k86r,
            ),
            rxn173 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII_RII production 53",
                reactants={BMP27_Alk3_RII_RII: 1, Alk8: 1},
                products={BMP27_Alk3_Alk8_RII_RII: 1},
                rate=k87,
            ),
            rxn174 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII_RII dissolution 54",
                reactants={BMP27_Alk3_Alk8_RII_RII: 1},
                products={BMP27_Alk3_RII_RII: 1, Alk8: 1},
                rate=k87r,
            ),
            rxn175 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII_RII production 55",
                reactants={BMP27_Alk8_Alk8_RII: 1, RII: 1},
                products={BMP27_Alk8_Alk8_RII_RII: 1},
                rate=k88,
            ),
            rxn176 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII_RII dissolution 56",
                reactants={BMP27_Alk8_Alk8_RII_RII: 1},
                products={BMP27_Alk8_Alk8_RII: 1, RII: 1},
                rate=k88r,
            ),
            rxn177 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII_RII production 57",
                reactants={BMP27_Alk8_RII_RII: 1, Alk3: 1},
                products={BMP27_Alk3_Alk8_RII_RII: 1},
                rate=k89,
            ),
            rxn178 := gillespy2.Reaction(
                name="BMP27_Alk3_Alk8_RII_RII dissolution 58",
                reactants={BMP27_Alk3_Alk8_RII_RII: 1},
                products={BMP27_Alk8_RII_RII: 1, Alk3: 1},
                rate=k89r,
            ),
            rxn179 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII_RII production 59",
                reactants={BMP27_Alk8_RII_RII: 1, Alk8: 1},
                products={BMP27_Alk8_Alk8_RII_RII: 1},
                rate=k90,
            ),
            rxn180 := gillespy2.Reaction(
                name="BMP27_Alk8_Alk8_RII_RII dissolution 60",
                reactants={BMP27_Alk8_Alk8_RII_RII: 1},
                products={BMP27_Alk8_RII_RII: 1, Alk8: 1},
                rate=k90r,
            ),
            rxn2001 := gillespy2.Reaction(
                name="endo271",
                reactants={BMP27_Alk3: 1},
                products={Alk3: 1},
                rate=k2000,
            ),
            rxn2002 := gillespy2.Reaction(
                name="endo272",
                reactants={BMP27_Alk8: 1},
                products={Alk8: 1},
                rate=k2000,
            ),
            rxn2003 := gillespy2.Reaction(
                name="endo273", reactants={BMP27_RII: 1}, products={RII: 1}, rate=k2000
            ),
            rxn2004 := gillespy2.Reaction(
                name="endo274",
                reactants={BMP27_Alk3_Alk3: 1},
                products={Alk3: 2},
                rate=k2000,
            ),
            rxn2005 := gillespy2.Reaction(
                name="endo275",
                reactants={BMP27_Alk3_Alk8: 1},
                products={Alk3: 1, Alk8: 1},
                rate=k2000,
            ),
            rxn2006 := gillespy2.Reaction(
                name="endo276",
                reactants={BMP27_Alk3_RII: 1},
                products={Alk3: 1, RII: 1},
                rate=k2000,
            ),
            rxn2007 := gillespy2.Reaction(
                name="endo277",
                reactants={BMP27_Alk8_Alk8: 1},
                products={Alk8: 2},
                rate=k2000,
            ),
            rxn2008 := gillespy2.Reaction(
                name="endo278",
                reactants={BMP27_Alk8_RII: 1},
                products={Alk8: 1, RII: 1},
                rate=k2000,
            ),
            rxn2009 := gillespy2.Reaction(
                name="endo279",
                reactants={BMP27_RII_RII: 1},
                products={RII: 2},
                rate=k2000,
            ),
            rxn2010 := gillespy2.Reaction(
                name="endo2710",
                reactants={BMP27_Alk3_Alk3_RII: 1},
                products={Alk3: 2, RII: 1},
                rate=k2000,
            ),
            rxn2011 := gillespy2.Reaction(
                name="endo2711",
                reactants={BMP27_Alk3_Alk8_RII: 1},
                products={Alk3: 1, Alk8: 1, RII: 1},
                rate=k2000,
            ),
            rxn2012 := gillespy2.Reaction(
                name="endo2712",
                reactants={BMP27_Alk3_RII_RII: 1},
                products={Alk3: 1, RII: 2},
                rate=k2000,
            ),
            rxn2013 := gillespy2.Reaction(
                name="endo2713",
                reactants={BMP27_Alk8_Alk8_RII: 1},
                products={Alk8: 2, RII: 1},
                rate=k2000,
            ),
            rxn2014 := gillespy2.Reaction(
                name="endo2714",
                reactants={BMP27_Alk8_RII_RII: 1},
                products={Alk8: 1, RII: 2},
                rate=k2000,
            ),
            rxn2015 := gillespy2.Reaction(
                name="endo2715",
                reactants={BMP27_Alk3_Alk3_RII_RII: 1},
                products={Alk3: 2, RII: 2},
                rate=k2000,
            ),
            rxn2016 := gillespy2.Reaction(
                name="endo2716",
                reactants={BMP27_Alk3_Alk8_RII_RII: 1},
                products={Alk3: 1, Alk8: 1, RII: 2},
                rate=k2000,
            ),
            rxn2017 := gillespy2.Reaction(
                name="endo2717",
                reactants={BMP27_Alk8_Alk8_RII_RII: 1},
                products={Alk8: 2, RII: 2},
                rate=k2000,
            ),
            rxn7001 := gillespy2.Reaction(
                name="endo71", reactants={BMP7_Alk3: 1}, products={Alk3: 1}, rate=k7000
            ),
            rxn7002 := gillespy2.Reaction(
                name="endo72", reactants={BMP7_Alk8: 1}, products={Alk8: 1}, rate=k7000
            ),
            rxn7003 := gillespy2.Reaction(
                name="endo73", reactants={BMP7_RII: 1}, products={RII: 1}, rate=k7000
            ),
            rxn7004 := gillespy2.Reaction(
                name="endo74",
                reactants={BMP7_Alk3_Alk3: 1},
                products={Alk3: 2},
                rate=k7000,
            ),
            rxn7005 := gillespy2.Reaction(
                name="endo75",
                reactants={BMP7_Alk3_Alk8: 1},
                products={Alk3: 1, Alk8: 1},
                rate=k7000,
            ),
            rxn7006 := gillespy2.Reaction(
                name="endo76",
                reactants={BMP7_Alk3_RII: 1},
                products={Alk3: 1, RII: 1},
                rate=k7000,
            ),
            rxn7007 := gillespy2.Reaction(
                name="endo77",
                reactants={BMP7_Alk8_Alk8: 1},
                products={Alk8: 2},
                rate=k7000,
            ),
            rxn7008 := gillespy2.Reaction(
                name="endo78",
                reactants={BMP7_Alk8_RII: 1},
                products={Alk8: 1, RII: 1},
                rate=k7000,
            ),
            rxn7009 := gillespy2.Reaction(
                name="endo79",
                reactants={BMP7_RII_RII: 1},
                products={RII: 2},
                rate=k7000,
            ),
            rxn7010 := gillespy2.Reaction(
                name="endo710",
                reactants={BMP7_Alk3_Alk3_RII: 1},
                products={Alk3: 2, RII: 1},
                rate=k7000,
            ),
            rxn7011 := gillespy2.Reaction(
                name="endo711",
                reactants={BMP7_Alk3_Alk8_RII: 1},
                products={Alk3: 1, Alk8: 1, RII: 1},
                rate=k7000,
            ),
            rxn7012 := gillespy2.Reaction(
                name="endo712",
                reactants={BMP7_Alk3_RII_RII: 1},
                products={Alk3: 1, RII: 2},
                rate=k7000,
            ),
            rxn7013 := gillespy2.Reaction(
                name="endo713",
                reactants={BMP7_Alk8_Alk8_RII: 1},
                products={Alk8: 2, RII: 1},
                rate=k7000,
            ),
            rxn7014 := gillespy2.Reaction(
                name="endo714",
                reactants={BMP7_Alk8_RII_RII: 1},
                products={Alk8: 1, RII: 2},
                rate=k7000,
            ),
            rxn7015 := gillespy2.Reaction(
                name="endo715",
                reactants={BMP7_Alk3_Alk3_RII_RII: 1},
                products={Alk3: 2, RII: 2},
                rate=k7000,
            ),
            rxn7016 := gillespy2.Reaction(
                name="endo716",
                reactants={BMP7_Alk3_Alk8_RII_RII: 1},
                products={Alk3: 1, Alk8: 1, RII: 2},
                rate=k7000,
            ),
            rxn7017 := gillespy2.Reaction(
                name="endo717",
                reactants={BMP7_Alk8_Alk8_RII_RII: 1},
                products={Alk8: 2, RII: 2},
                rate=k7000,
            ),
            rxn1001 := gillespy2.Reaction(
                name="endo1", reactants={BMP2_Alk3: 1}, products={Alk3: 1}, rate=k1000
            ),
            rxn1002 := gillespy2.Reaction(
                name="endo2", reactants={BMP2_Alk8: 1}, products={Alk8: 1}, rate=k1000
            ),
            rxn1003 := gillespy2.Reaction(
                name="endo3", reactants={BMP2_RII: 1}, products={RII: 1}, rate=k1000
            ),
            rxn1004 := gillespy2.Reaction(
                name="endo4",
                reactants={BMP2_Alk3_Alk3: 1},
                products={Alk3: 2},
                rate=k1000,
            ),
            rxn1005 := gillespy2.Reaction(
                name="endo5",
                reactants={BMP2_Alk3_Alk8: 1},
                products={Alk3: 1, Alk8: 1},
                rate=k1000,
            ),
            rxn1006 := gillespy2.Reaction(
                name="endo6",
                reactants={BMP2_Alk3_RII: 1},
                products={Alk3: 1, RII: 1},
                rate=k1000,
            ),
            rxn1007 := gillespy2.Reaction(
                name="endo7",
                reactants={BMP2_Alk8_Alk8: 1},
                products={Alk8: 2},
                rate=k1000,
            ),
            rxn1008 := gillespy2.Reaction(
                name="endo8",
                reactants={BMP2_Alk8_RII: 1},
                products={Alk8: 1, RII: 1},
                rate=k1000,
            ),
            rxn1009 := gillespy2.Reaction(
                name="endo9", reactants={BMP2_RII_RII: 1}, products={RII: 2}, rate=k1000
            ),
            rxn1010 := gillespy2.Reaction(
                name="endo10",
                reactants={BMP2_Alk3_Alk3_RII: 1},
                products={Alk3: 2, RII: 1},
                rate=k1000,
            ),
            rxn1011 := gillespy2.Reaction(
                name="endo11",
                reactants={BMP2_Alk3_Alk8_RII: 1},
                products={Alk3: 1, Alk8: 1, RII: 1},
                rate=k1000,
            ),
            rxn1012 := gillespy2.Reaction(
                name="endo12",
                reactants={BMP2_Alk3_RII_RII: 1},
                products={Alk3: 1, RII: 2},
                rate=k1000,
            ),
            rxn1013 := gillespy2.Reaction(
                name="endo13",
                reactants={BMP2_Alk8_Alk8_RII: 1},
                products={Alk8: 2, RII: 1},
                rate=k1000,
            ),
            rxn1014 := gillespy2.Reaction(
                name="endo14",
                reactants={BMP2_Alk8_RII_RII: 1},
                products={Alk8: 1, RII: 2},
                rate=k1000,
            ),
            rxn1015 := gillespy2.Reaction(
                name="endo15",
                reactants={BMP2_Alk3_Alk3_RII_RII: 1},
                products={Alk3: 2, RII: 2},
                rate=k1000,
            ),
            rxn1016 := gillespy2.Reaction(
                name="endo16",
                reactants={BMP2_Alk3_Alk8_RII_RII: 1},
                products={Alk3: 1, Alk8: 1, RII: 2},
                rate=k1000,
            ),
            rxn1017 := gillespy2.Reaction(
                name="endo17",
                reactants={BMP2_Alk8_Alk8_RII_RII: 1},
                products={Alk8: 2, RII: 2},
                rate=k1000,
            ),
        ]
    )

    model.timespan(parameter_values.timespan)
    return model
