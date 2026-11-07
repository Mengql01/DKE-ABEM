"""
TMF Knowledge Base Tools for TRNSYS DCK Generator.

Provides @tool decorated functions for querying TMF component specifications.
These tools are used by all methods (B1, B2, B3, Proposed) for component validation.

Tool Categories:
1. Component Specification Tools
   - get_tmf_spec: Get full TMF specification for a component
   - get_component_io_mapping: Get input/output mapping for connections

2. Validation Tools
   - validate_parameter_count: Validate parameter count matches TMF spec
   - validate_input_count: Validate input count matches TMF spec

Data Source:
- 765 TMF structured parsing files from training data
- Located at: settings.tmf_parsed_path

Special Handling:
- Type56 (Multi-Zone Building): Requires external .b18/.b17/.bui file to get complete I/O info
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# === TMF Data Cache ===

_tmf_data_cache: Optional[Dict[str, Any]] = None

# === Flexible Input Components (Category C) ===
# Must stay in sync with Category C list in connection_generator.py system prompt.
FLEXIBLE_INPUT_TYPES = {
    # Ground/Slab
    "Type49", "Type1244", "Type1255", "Type1256", "Type957",
    # Storage Tank
    "Type158", "Type531", "Type533", "Type534", "Type541d", "Type550_IAM_Mode_2",
    # HVAC/AHU
    "Type151", "Type152",
    # Valve/Mixer
    "Type594", "Type595", "Type646", "Type647", "Type648", "Type649", "Type1260",
    # Plotter/Print
    "Type24", "Type25b", "Type25c", "Type27c", "Type28b",
    "Type46a", "Type46b", "Type65a", "Type65c", "Type65d",
    # Controller
    "Type40", "Type150", "Type1502", "Type1503", "Type970", "Type971",
    "Type1250", "Type698",
    # Building/Zone
    "Type660", "Type690", "Type759", "Type932", "Type2280",
    # EES/Matlab/Excel
    "Type62", "Type66a", "Type66b", "Type66c", "Type66d", "Type155",
    # Solver
    "Type78a",
    # Coupling
    "Type97",
    # Other
    "Type39", "Type510",
}

# === Type56 Special Handling ===
# Type56 (Multi-Zone Building) requires external .b18/.b17/.bui file to get complete I/O info

_INPUTS_LINE_RE = re.compile(r"^\s*INPUTS(?:\s+(.+))?\s*$", re.IGNORECASE)
_INPUTS_DESC_HEADER_RE = re.compile(r"^\s*INPUTS_DESCRIPTION\s*$", re.IGNORECASE)
_INPUTS_DESC_LINE_RE = re.compile(r"^\s*([^:]+?)\s*:\s*([^:]*?)\s*:\s*(.*?)\s*$")
_OUTPUTS_HEADER_RE = re.compile(r"^\s*OUTPUTS\s*$", re.IGNORECASE)
_AIRNODES_RE = re.compile(r"^\s*AIRNODES\s*=\s*(.+?)\s*$", re.IGNORECASE)
_NTYPES_RE = re.compile(r"^\s*NTYPES\s*=\s*(.+?)\s*$", re.IGNORECASE)
_NTYPES_CONT_RE = re.compile(r"^\s*=\s*(.+?)\s*$")
_OUTPUT_TYPE_ENTRY_RE = re.compile(r"^\s*(\d+)\s*:\s*([A-Za-z0-9_]+)\s*-\s*(.+?)\s*$")
_OUTPUT_SURF_ENTRY_RE = re.compile(
    r"^\s*(\d+)\s*:\s*SURF\s*=\s*(.*?)\s*:\s*([A-Za-z0-9_]+)\s*-\s*(.+?)\s*$",
    re.IGNORECASE,
)

# Type56 template JSON (complete version from get_tmf-type56_spec.py)
_TYPE56_TEMPLATE_JSON = r"""{
    "Type56.tmf": {
      "tmf_path": "d:\\TRNSYS18_03\\Studio\\Proformas\\Loads and Structures\\Multi-Zone Building\\Type56.tmf",
      "format_marker": "#ESI!",
      "version": "4.0",
      "model_name": "Multi-Zone Building",
      "type_number": 56,
      "author": {
        "name": "Transsolar Energietechnik GmbH, Stuttgart",
        "organization": "Transsolar Energietechnik GmbH, Stuttgart"
      },
      "version_info": {
        "version": "1",
        "modification_date": "TRNSYS v18",
        "modifier": "MH",
        "last_update": "2017-01-31"
      },
      "description": "This component models the thermal behavior of a building having multiple thermal zones.\n\nThe building description is passed to the model by the building description file (*.b18,*.b17, *.bui).",
      "source_code_path": "",
      "variable_counts": {
        "parameters": 3,
        "inputs": 7,
        "outputs": 1,
        "total": 5
      },
      "parameters": [
        {
          "index": 1,
          "name": "Logical unit for building description file (*.b18, *.b17, *.bui)",
          "description": "The logical unit through which the building description file will be read. Each external file that TRNSYS reads from or writes to must be assigned a unique logical unit number in the TRNSYS input file. The building description file is created by the TRNBuild program and is typically called *.b17 and *.bui, respectively.",
          "data_type": "integer",
          "io_type": 3,
          "default_value": "30",
          "unit": null,
          "dimension": null,
          "min_value": "30",
          "max_value": "999",
          "category": "simple"
        },
        {
          "index": 2,
          "name": "Star network calculation switch",
          "description": "This parameter indicates whether the star network calculations should be performed only at the beginning of the simulation (=0) or at every timestep (=1).  For most simulations, the star network can be calculated just at the beginning of the simulation.  However, if the convective heat transfer coefficient at the inside surface of any of the walls in the building is variable (not a constant value), this parameter should be set to 1.",
          "data_type": "integer",
          "io_type": 3,
          "default_value": "0",
          "unit": null,
          "dimension": null,
          "min_value": "0",
          "max_value": "100",
          "category": "simple"
        },
        {
          "index": 3,
          "name": "Weighting factor for operative temperature",
          "description": "The weighting factor for the operative room temperature.  The operative room temperature is a function of both the air and surface temperatures in the zone: Top = Aop*Tair + (1-Aop)*Tsurf (See manual for further information on equation) This temperature is used most often in room comfort analysis simulations.",
          "data_type": "real",
          "io_type": 3,
          "default_value": "0.50",
          "unit": null,
          "dimension": null,
          "min_value": "0.0",
          "max_value": "1.0",
          "category": "simple"
        }
      ],
      "inputs": [
        {
          "index": 1,
          "name": " 1- TAMB",
          "description": "ambient temperature",
          "data_type": "real",
          "io_type": 1,
          "default_value": "0",
          "unit": "C",
          "dimension": "Temperature",
          "min_value": "-Inf",
          "max_value": "+Inf",
          "category": "simple"
        },
        {
          "index": 2,
          "name": " 2- RELHUMAMB",
          "description": "relative ambient humidity",
          "data_type": "real",
          "io_type": 1,
          "default_value": "0",
          "unit": "%",
          "dimension": "any",
          "min_value": "-Inf",
          "max_value": "+Inf",
          "category": "simple"
        },
        {
          "index": 3,
          "name": " 3- TSKY",
          "description": "effective sky temperature for longwave radiation exchange (FSKY)",
          "data_type": "real",
          "io_type": 1,
          "default_value": "0",
          "unit": "C",
          "dimension": "Temperature",
          "min_value": "-Inf",
          "max_value": "+Inf",
          "category": "simple"
        },
        {
          "index": 4,
          "name": " 4- TSGRD",
          "description": "effective ground temperature for longwave radiation exchange (1-FSKY)",
          "data_type": "real",
          "io_type": 1,
          "default_value": "0",
          "unit": "C",
          "dimension": "Temperature",
          "min_value": "-Inf",
          "max_value": "+Inf",
          "category": "simple"
        },
        {
          "index": 5,
          "name": " 5- AZEN",
          "description": "solar zenith angle",
          "data_type": "real",
          "io_type": 1,
          "default_value": "0",
          "unit": "degrees",
          "dimension": "Direction (Angle)",
          "min_value": "-Inf",
          "max_value": "+Inf",
          "category": "simple"
        },
        {
          "index": 6,
          "name": " 6- AAZM",
          "description": "solar azimuth angle (0=facing equator, 90=facing west, -90=270=facing east)",
          "data_type": "real",
          "io_type": 1,
          "default_value": "0",
          "unit": "degrees",
          "dimension": "Direction (Angle)",
          "min_value": "-Inf",
          "max_value": "+Inf",
          "category": "simple"
        },
        {
          "index": 7,
          "name": " 7- GRDREF",
          "description": "ground reflectance (range 0...1)",
          "data_type": "real",
          "io_type": 1,
          "default_value": "0",
          "unit": "any",
          "dimension": "any",
          "min_value": "-Inf",
          "max_value": "+Inf",
          "category": "simple"
        }
      ],
      "external_files": [],
      "tags": [],
      "parse_info": {
        "var_start_line": 51,
        "declared_var_count": 5,
        "skipped_blocks": 0,
        "parser_version": "v7.1_smart_adaptive"
      }
    }
}"""

_TYPE56_TEMPLATE: Dict[str, Any] = json.loads(_TYPE56_TEMPLATE_JSON)

# Type56 optional outputs CSV data (for unit lookup)
_TRNSYS_TYPE56_OUTPUTS_CSV = """Category,NType#,Label,Description,Unit
Airnodes Outputs:,1,TAIR,air temperature of airnode,[°C]
Airnodes Outputs:,2,QSENS,"sensible energy demand, heating(-), cooling(+)(supply air conditioning not included)",[kJ/hr]
Airnodes Outputs:,3,QCSURF,total convection to air from all surfaces within airnode incl. internal shading,[kJ/hr]
Airnodes Outputs:,4,QINF,sensible infiltration energy gain of airnode,[kJ/hr]
Airnodes Outputs:,5,QVENT,sensible ventilation energy gain of airnode,[kJ/hr]
Airnodes Outputs:,6,QCOUP,sensible coupling gains of airnode,[kJ/hr]
Airnodes Outputs:,7,QGCONV,internal convective gains of airnode,[kJ/hr]
Airnodes Outputs:,8,DQAIR,sensible change in internal energy of air in airnode since the beginning of the simulation,[kJ/hr]
Airnodes Outputs:,9,RELHUM,relative humidity of airnode air,[%]
Airnodes Outputs:,10,QLATD,"latent energy demand of airnode, humidification(-), dehumidification (+); (supply air conditioning not included)",[kJ/hr]
Airnodes Outputs:,11,QLATG,"latent energy gains of airnode including ventilation, infiltration, coupling, internal latent gains and vapor adsorption in walls",[kJ/hr]
Airnodes Outputs:,12,QSOLTR,total shortwave solar radiation transmitted through external windows of airnode (but not kept 100 % in airnode),[kJ/hr]
Airnodes Outputs:,13,QGRAD,defined internal radiative gains of airnode,[kJ/hr]
Airnodes Outputs:,14,QTABSI,"total radiation absorbed (or transmitted) at all inside surf. of airnode (includes solar gains, radiative heat, internal radiative gains and wallgains)",[kJ/hr]
Airnodes Outputs:,15,QTABSO,"total radiation absorbed at all outside surf. of airnode (includes solar gains, radiative heat, internal radiative gains and wallgains, but not longwave radiation exchange with Tsky or other surfaces)",[kJ/hr]
Airnodes Outputs:,16,QTCOMO,total convective and longwave rad. gains (Tsky) to outside surf.,[kJ/hr]
Surface Outputs:,17,TSI,inside surface temperature,[°C]
Surface Outputs:,18,TSO,outside surface temperature,[°C]
Surface Outputs:,19,QCOMI,energy from inside surf. incl. conv. to air and longwave radiation to other surfaces,[kJ/hr]
Surface Outputs:,20,QCOMO,energy to outside surf. incl. conv. to air and longwave radiation to other surfaces or Tsky,[kJ/hr]
Surface Outputs:,21,QABSI,"absorbed (or transmitted) at inside surf (includes solar gains, radiative heat, internal radiative gains and wallgains, except longwave radiation exchange with other walls)",[kJ/hr]
Surface Outputs:,22,QABSO,"radiation absorbed at outside surf. [kJ/hr] (includes solar gains, radiative heat, internal radiative gains and wallgains, except longwave radiation exchange with other walls or Tsky)",[kJ/hr]
Airnode Outputs:,23,TSTAR,star node temperature of airnode,[°C]
Airnode Outputs:,24,TMSURF,weighted mean surface temperature of airnode,[°C]
Airnode Outputs:,25,TOP,operative airnode temperature,[°C]
Airnode Outputs:,26,QVAPW,heat of vapor adsorption in walls of airnode,[kJ/hr]
Airnode Outputs:,27,QUA,static UA-transmission losses (UA*dT) of airnode,[kJ/hr]
Airnode Outputs:,28,value of schedule,value of schedule,
Airnode Outputs:,29,ABSHUM,absolute humidity of airnode air,[kgwater / kgdry_air]
Airnode Outputs:,30,QHEAT,sensible heating demand of airnode (positive values); (supply air conditioning not included),[kJ/hr]
Airnode Outputs:,31,QCOOL,sensible cooling demand of airnode (positive values); (supply air conditioning not included),[kJ/hr]
Outputs for Groups of Airnodes:,32,SQHEAT,sum of sensible heating demand for specified airnodes (positive); (supply air conditioning not included),[kJ/hr]
Outputs for Groups of Airnodes:,33,SQCOOL,sum of sensible cooling demand for specified airnodes (positive); (supply air conditioning not included),[kJ/hr]
Outputs for Groups of Airnodes:,34,SQCSURF,sum of surf. conv. gains of specified airnodes,[kJ/hr]
Outputs for Groups of Airnodes:,35,SQINF,sum of sensible infiltration gains of specified airnodes,[kJ/hr]
Outputs for Groups of Airnodes:,36,SQVENT,sum of sensible ventilation gains of specified airnodes,[kJ/hr]
Outputs for Groups of Airnodes:,37,SQCOUP,sum of sensible coupling gains of specified airnodes,[kJ/hr]
Outputs for Groups of Airnodes:,38,SQGCONV,sum of int. conv. gains of specified airnodes,[kJ/hr]
Outputs for Groups of Airnodes:,39,SDQAIR,sum of changes in internal energy of air in specified airnodes (since the beginning of the simulation),[kJ/hr]
Outputs for Groups of Airnodes:,40,SQLATD,"sum of latent energy demand of specified airnodes humidification(-), dehumidification (+);(supply air conditioning not included)",[kJ/hr]
Outputs for Groups of Airnodes:,41,SQLATG,"sum of latent energy gains of specified airnodes including ventilation, infiltration, coupling and vapor adsorption in walls",[kJ/hr]
Outputs for Groups of Airnodes:,42,SQSOLT,sum of shortwave solar radiation transmitted through windows of specified airnodes (but not kept 100 % in airnode),[kJ/hr]
Outputs for Groups of Airnodes:,43,SQGRAD,sum of internal radiative gains of specified airnodes,[kJ/hr]
Outputs for Groups of Airnodes:,44,SQABSI,"total rad. absorbed (or transmitted) at inside surf. of specified airnodes (includes solar gains, rad. heat, int. rad. and wallgains)",[kJ/hr]
Outputs for Groups of Airnodes:,45,SQABSO,"total rad. absorbed at outside surf. of specified airnodes (incl. solar gains, rad. heat, int. rad. and wallgains, but not l-wave with Tsky)",[kJ/hr]
Outputs for Groups of Airnodes:,46,SQUA,sum of static transmission losses (UA*dT) of specified airnodes,[kJ/hr]
Outputs for Groups of Airnodes:,47,SQVAPW,sum of heat of vapor adsorption in walls of specified airnodes,[kJ/hr]
Surface Outputs:,48,ICOND,condensation flag (0 or 1) for inside surfaces,
Surface Outputs:,49,OCOND,condensation flag (0 or 1) for outside surfaces,
Surface Outputs:,50,UWIN,U-value of glazing and frame for external windows,[kJ/ hr m² K]
Surface Outputs:,51,GWIN,g-value (solar heat gain coeff.) of glazing of external window only (= Balance 3 (primary solar): gglass,
Surface Outputs:,52,TIGLS,inside surface temperature of the glazing,[°C]
Surface Outputs:,53,TOGLS,outside surface temperature of the glazing,[°C]
Surface Outputs:,54,TIFRM,inside surface temperature of the frame,[°C]
Surface Outputs:,55,TOFRM,outside surface temperature of the frame,[°C]
Airnode Outputs:,56,QSEC,secondary heat flux of all windows of airnode,[kJ/hr]
Surface Outputs:,57,TALM,node temperature of active layer,[°C]
Surface Outputs:,58,TOFL,fluid outlet temperature of active layer,[°C]
Surface Outputs:,59,QALFL,"energy input by fluid of active layer to active layer(> 0: cooling, < 0: heating)",[kJ/hr]
Surface Outputs:,60,QALE,energy input by gains of active layer to active layer,[kJ/hr]
Surface Outputs:,61,QALTL,total energy input by fluid&gains of active layer to active layer,[kJ/hr]
Airnode Outputs:,62,PMV,predicted mean vote (PMV) value of airnode,
Airnode Outputs:,63,PPD,predicted percentage of dissatisfied persons (PPD) value of airnode,[%]
Surface Outputs:,64,QSGL,solar rad. absorbed on all panes of window,[kJ/hr]
Surface Outputs:,65,QSISH,solar rad. absorbed on internal shading device of window,[kJ/hr]
Surface Outputs:,66,QSOFR,solar rad. absorbed on outside of window frame,[kJ/hr]
Surface Outputs:,67,QSIFR,solar rad. absorbed on inside of window frame,[kJ/hr]
Surface Outputs:,68,QSOUT,solar transmission to outside through external window,[kJ/hr]
Airnode Outputs:,69,QTSGL,total solar rad. absorbed on all panes of all windows of airnode,[kJ/hr]
Airnode Outputs:,70,QTSISH,total solar absorbed on internal shading device of all windows of airnode,[kJ/hr]
Airnode Outputs:,71,QTSOFR,total solar absorption on outside frame of external windows of airnode,[kJ/hr]
Airnode Outputs:,72,QTSIFR,total solar rad. absorbed on inside of the frame of all ext. window and both sides of all adjacent windows of airnode,[kJ/hr]
Airnode Outputs:,73,QTSOUT,total solar transmission to outside through external window of airnode,[kJ/hr]
Airnode Outputs:,74,QTSPAS,total solar radiation passing the glass surface of external windows (absorption on external shading devices and reflection of external glass surface are excluded! -> total radiation absorbed or transmitted by building components),[kJ/hr]
Airnode Outputs:,75,QTSABS,total solar rad. absorbed at all inside surfaces of airnode (for windows: secondary heat flux),[kJ/hr]
Airnode Outputs:,76,QTWG,total wallgains on inside surfaces of airnode,[kJ/hr]
Airnode Outputs:,77,QTSKY,total longwave rad. losses to sky of outside surfaces of airnode,[kJ/hr]
Airnode Outputs:,78,QRHEAT,radiative energy rate of sensible heating demand of airnode,[kJ/hr]
Surface Outputs:,79,QSIAB,Total solar (direct & diffuse) rad. absorbed at inside surface; for windows: secondary heat flux,[kJ/hr]
Surface Outputs:,80,QIBAB,solar direct rad. absorbed at inside surface,[kJ/hr]
Surface Outputs:,81,QIDAB,solar diffuse rad. being absorbed at inside surface,[kJ/hr]
Surface Outputs:,82,QWG,wall gain on inside surface of wall or window,[kJ/hr]
Surface Outputs:,83,QSKY,longwave rad. losses to sky of external surface,[kJ/hr]
Surface Outputs:,84,QRGAB,internal rad gains absorbed on inside surface,[kJ/hr]
Surface Outputs:,85,QRHEA,rad. energy rate of sens. heating demand absorbed on inside surf,[kJ/hr]
Surface Outputs:,86,REQV,equivalent resistance between star node and surface,[(hr K)/kJ]
Airnode Outputs:,90,THEAT,Set temperature heating,[°C]
Airnode Outputs:,91,PHMAX,Maximum power of heating,[kJ/hr]
Airnode Outputs:,92,HUMHEAT,Desired Humidity for heating,[%] or [kg/kg]
Airnode Outputs:,93,TCOOL,Set temperature cooling,[°C]
Airnode Outputs:,94,PCMAX,Maximum power of cooling,[kJ/hr]
Airnode Outputs:,95,HUMCOOL,Desired Humidity for cooling,[%] or [kg/kg]
Airnode Outputs:,96,ACHINF,Airchange rate infiltration,[1/h]
Airnode Outputs:,97,GABSHUM,humidity gain of defined gains,[kg/kg]
Airnode Outputs:,98,QDEHUM,latent energy demand of airnode by dehumidification (positive values),[kJ/hr]
Airnode Outputs:,99,QHUM,latent energy demand of airnode by humidification (positive values),[kJ/hr]
Surface Outputs:,100,ISHADE,Internal shading factor of window,[-]
Surface Outputs:,101,OSHADB,Total external shading factor of window for direct radiation incl. geometric shading,[-]
Surface Outputs:,102,OSHADD,Total external shading factor of window for diffuse radiation incl. geometric shading,[-]
Surface Outputs:,103,IBSHAD,Total direct radiation on outside of external surface (incl. shading),[kJ/hr]
Surface Outputs:,104,IDSHAD,Total diffuse radiation on outside of external surface (incl. shading),[kJ/hr]
Surface Outputs:,105,FSOLBPR,Primary beam radiation distribution factor of inside surface,[-]
Surface Outputs:,106,FSOLB,Non-primary beam radiation distribution factor of inside surface,[-]
Surface Outputs:,107,HCONVO,eff. outside conv. heat transfer coeff (BACK),[kJ/hr m² K]
Surface Outputs:,108,HCONVI,eff. inside conv. heat transfer coeff (FRONT),[kJ/hr m² K]
Surface Outputs:,109,QSICONV,Energy gain inside by convection,[kJ/hr]
Surface Outputs:,110,QABSILW,Energy absorbed on inside surface by longwave radiation exchange,[kJ/hr]
Surface Outputs:,111,TIFL,Inlet temperature of active layer,[°C]
Surface Outputs:,112,MFLAL,Inlet mass flow rate of active layer,[kg/hr]
Surface Outputs:,113,AREA,surface area,[m²]
Surface Outputs:,114,IB,Incident direct radiation on outside of external surfaces (without shading effects),[kJ/hr]
Surface Outputs:,115,ID,Incident diffuse radiation on outside of external surfaces (without shading effects),[kJ/hr]
Surface Outputs:,116,IT,Incident total radiation on outside of external surfaces (without shading effects),[kJ/hr]
Airnode Outputs:,117,QSOLAIR,convective energy gain of airnode due transmitted solar radiation through external windows which is transformed immediately into a convective heat flow,[kJ/hr]
Airnode Outputs:,118,VOLUME,Air volume of airnode,[m³]
Surface Outputs:,119,SHADCNTRL,"Shading control signal for integrated radiation depending shading of external windows (0..integrated shading control not active, 1.. integrated shading control active (=closed))",[-]
Comfort Outputs:,120,TMR,Mean radiant temperature (output not influenced by elevated air speed),[°C]
Comfort Outputs:,122,TOP,Operative temperature,[°C]
Comfort Outputs:,123,SET,Standard effective temperature,[°C]
Comfort Outputs:,124,PMV,Predicted mean vote (incl. elevated air speed),[-]
Comfort Outputs:,126,PPD,Percentage of persons dissatisfied (incl. elevated air speed),[%]
Comfort Outputs:,128,CLO,Clothing factor,[clo]
Comfort Outputs:,129,MET,Metabolic rate,[met]
Comfort Outputs:,130,WORK,External work,[met]
Comfort Outputs:,131,VAIR,Air velocity (optimized elevated air speed if active),[m/s]
Comfort Outputs:,132,EAS,"Optimized elevated air speed level (0,1,2 or 3)",[m/s]
Comfort Outputs:,133,FSHADE,Shading factor of comfort sensor,[-]
Comfort Outputs:,134,FPROJ,"Sun position depending projection factor (0…no, 1…yes)",[-]
Surface Outputs:,140,GSHADB,Geometric shading factor of window for direct radiation,[-]
Surface Outputs:,141,GSHADD,Geometric shading factor of window for diffuse radiation,[-]
Airnode Outputs:,150,REFAREA,Reference floor area,[m²]
Airnode Outputs:,151,QGLIGHT,Sensible energy gains from “lights” gain of airnode,[kJ/hr]
Airnode Outputs:,152,QGEQUIP,Sensible energy gains from “equipment” gains of airnode,[kJ/hr]
Airnode Outputs:,153,QGMISC,Sensible energy gains from “miscellaneous” gains of airnode,[kJ/hr]
Airnode Outputs:,154,QGPEOPLE,Sensible energy gains from “people” gains of airnode,[kJ/hr]
Airnode Outputs:,155,QELLIGHT,Electric energy demand of “lights” gains of airnode,[kJ/hr]
Airnode Outputs:,156,QELEQUIP,Electric energy demand of “equipment” gains of airnode,[kJ/hr]
Airnode Outputs:,157,QELMISC,Electric energy demand of “miscellaneous” gains of airnode,[kJ/hr]
Airnode Outputs:,158,QELGAINS,"Electric energy demand of ""all"" gains of airnode",[kJ/hr]
Airnode Outputs:,159,QELHEAT,Electric energy demand of heating of airnode,[kJ/hr]
Airnode Outputs:,160,QELCOOL,Electric energy demand of cooling of airnode,[kJ/hr]
Airnode Outputs:,161,QELVENT,Electric energy demand of ventilation of airnode,[kJ/hr]
Airnode Outputs:,162,QELHVAC,"Electric energy demand of HVAC components (heating, cooling, ventilation) of airnode",[kJ/hr]
Airnode Outputs:,163,THBAREA,Reference area for thermal bridge gain/loss of airnode,[C]
Airnode Outputs:,164,QGTHB,"Sensible energy gain from ""thermal bridge"" gains/losses of airnode",[kJ/hr]
Airnode Outputs:,170,QAHU_H,total heating energy to condition supply air to set point conditions (sum of defined ventilations with supply air conditioning),[kJ/hr]
Airnode Outputs:,171,QAHU_C,total cooling energy to condition supply air to set point conditions (sum of defined ventilations with supply air conditioning),[kJ/hr]
Airnode Outputs:,172,QAHU_SENS_H,"heating energy to condition supply air to reach set point temperature after HX, Dehum (sum of defined ventilations with supply air conditioning)",[kJ/hr]
Airnode Outputs:,173,QAHU_SENS_C,cooling energy to condition supply air to reach set point temperature after HX (sum of defined ventilations with supply air conditioning),[kJ/hr]
Airnode Outputs:,174,QAHU_HRSENS_H,heating energy provides by heat recovery unit (sum of defined ventilations with supply air conditioning),[kJ/hr]
Airnode Outputs:,175,QAHU_HRSENS_C,cooling energy provides by heat recovery unit (sum of defined ventilations with supply air conditioning),[kJ/hr]
Airnode Outputs:,176,QAHU_HUM_H,heating energy for humidification process (sum of defined ventilations with supply air conditioning),[kJ/hr]
Airnode Outputs:,177,QAHU_DEHUM_C,cooling energy for dehumidification process (sum of defined ventilations with supply air conditioning),[kJ/hr]
Airnode Outputs:,180,TAIR_WB,wet bulb temperature of airnode,[C]
Airnode Outputs:,181,TAIR_DP,dew point temperature of airnode,[C]
Airnode Outputs:,185,TAHU_HR,dry bulb temperature after heat recovery of 1st ventilation with supply air conditioning,[C]
Airnode Outputs:,186,TAHU_HRWB,wet bulb temperature after heat recovery of 1st ventilation with supply air conditioning,[C]
Airnode Outputs:,187,TAHU_HRDP,dew point temperature after heat recovery of 1st ventilation with supply air conditioning,[C]
Airnode Outputs:,188,RHAHU_HR,rel. humidity of air after heat recovery of 1st ventilation with supply air conditioning,[%]
Airnode Outputs:,189,WHU_HR,Humidity ratio of air after heat recovery of 1st ventilation with supply air conditioning,[kg/kg]
Surface Outputs (complex fenestration model):,301,QSBSH,Solar rad. absorbed on shading devices between of glass panes (for complex fenestration model only),[kJ/h]
Surface Outputs (complex fenestration model):,302,TGLAY1,Mean temperature of 1st glazing layer (available for complex fenestration model only),[°C]
Surface Outputs (complex fenestration model):,303,TGLAY2,Mean temperature of 2nd glazing layer (available for complex fenestration model only),[°C]
Surface Outputs (complex fenestration model):,304,TGLAY3,Mean temperature of 3rd glazing layer (available for complex fenestration model only),[°C]
Surface Outputs (complex fenestration model):,305,TGLAY4,Mean temperature of 4th glazing layer (available for complex fenestration model only),[°C]
Surface Outputs (complex fenestration model):,306,TGLAY5,Mean temperature of 5th glazing layer (available for complex fenestration model only),[°C]
Surface Outputs (complex fenestration model):,307,TGLAY6,Mean temperature of 6th glazing layer (available for complex fenestration model only),[°C]
Surface Outputs (complex fenestration model):,310,TGGAP1,"Outlet temperature of gap 1 (for not ventilated gaps = gap temperature, available for complex fenestration model only)",[°C]
Surface Outputs (complex fenestration model):,311,TGGAP2,"Outlet temperature of gap 2 (for not ventilated gaps = gap temperature, available for complex fenestration model only)",[°C]
Surface Outputs (complex fenestration model):,312,TGGAP3,"Outlet temperature of gap 3 (for not ventilated gaps = gap temperature, available for complex fenestration model only)",[°C]
Surface Outputs (complex fenestration model):,313,TGGAP4,"Outlet temperature of gap 4 (for not ventilated gaps = gap temperature, available for complex fenestration model only)",[°C]
Surface Outputs (complex fenestration model):,314,TGGAP5,"Outlet temperature of gap 5 (for not ventilated gaps = gap temperature, available for complex fenestration model only)",[°C]
Surface Outputs (complex fenestration model):,317,MFGAP1,mass flow of gap 1 (available for complex fenestration model only),[kg/hr]
Surface Outputs (complex fenestration model):,318,MFGAP2,mass flow of gap 2 (available for complex fenestration model only),[kg/hr]
Surface Outputs (complex fenestration model):,319,MFGAP3,mass flow of gap 3 (available for complex fenestration model only),[kg/hr]
Surface Outputs (complex fenestration model):,320,MFGAP4,mass flow of gap 4 (available for complex fenestration model only),[kg/hr]
Surface Outputs (complex fenestration model):,321,MFGAP5,mass flow of gap 5 (available for complex fenestration model only),[kg/hr]
Airnode Outputs (Daylight):,400,ILLUM_H,Global horizontal illuminance outside the building (without shading effects),[lux]
Airnode Outputs (Daylight):,401,ILLUM_DH,Diffuse horizontal illuminance outside the building (without shading effects),[lux]
Airnode Outputs (Daylight):,402,ILLUM_DN,Direct normal illuminance outside the building (without shading effects),[lux]
Airnode Outputs (Daylight):,403,DLSHADE,Daylight shading control signal,[-]
Airnode Outputs (Daylight):,405,AREADCL1,Floor area related to 1st daylight controlled gain,[m²]
Airnode Outputs (Daylight):,406,ILLUMDLC1,Daylight illuminance of 1st controlled gain,[lux]
Airnode Outputs (Daylight):,407,ILLSETDLC1,Illuminance set point of 1st daylight controlled gain,[lux]
Airnode Outputs (Daylight):,408,TYPEDLC1,"Lighting control type of 1st daylight controlled gain (1…always on, 2…on/off, 3…continuous, 4…continuous on/off)",[-]
Airnode Outputs (Daylight):,409,FDIMDLC1,Dimming fraction of 1st daylight controlled gain,[-]
Airnode Outputs (Daylight):,410,CDADLC1,Continuous daylight autonomy of 1st daylight controlled gain,[%]
Airnode Outputs (Daylight):,411,DADLC1,Daylight autonomy of 1st daylight controlled gain,[%]
Airnode Outputs (Daylight):,412,UDIDLC1,Useful daylight illuminance of 1st daylight controlled gain,[%]
Airnode Outputs (Daylight):,415,AREADCL2,Floor area related to 2nd daylight controlled gain,[m²]
Airnode Outputs (Daylight):,416,ILLUMDLC2,Daylight illuminance of 2nd controlled gain,[lux]
Airnode Outputs (Daylight):,417,ILLSETDLC2,Illuminance set point of 2nd daylight controlled gain,[lux]
Airnode Outputs (Daylight):,418,TYPEDLC2,"Lighting control type of 2nd daylight controlled gain (1…always on, 2…on/off, 3…continuous, 4…continuous on/off)",[-]
Airnode Outputs (Daylight):,419,FDIMDLC2,Dimming fraction of 2nd daylight controlled gain,[-]
Airnode Outputs (Daylight):,420,CDADLC2,Continuous daylight autonomy of 2nd daylight controlled gain,[%]
Airnode Outputs (Daylight):,421,DADLC2,Daylight autonomy of 2nd daylight controlled gain,[%]
Airnode Outputs (Daylight):,422,UDIDLC2,Useful daylight illuminance of 2nd daylight controlled gain,[%]
Airnode Outputs (Daylight):,425,AREADCL3,Floor area related to 3rd daylight controlled gain,[m²]
Airnode Outputs (Daylight):,426,ILLUMDLC3,Daylight illuminance of 3rd controlled gain,[lux]
Airnode Outputs (Daylight):,427,ILLSETDLC3,Illuminance set point of 3rd daylight controlled gain,[lux]
Airnode Outputs (Daylight):,428,TYPEDLC3,"Lighting control type of 3rd daylight controlled gain (1…always on, 2…on/off, 3…continuous, 4…continuous on/off)",[-]
Airnode Outputs (Daylight):,429,FDIMDLC3,Dimming fraction of 3rd daylight controlled gain,[-]
Airnode Outputs (Daylight):,430,CDADLC3,Continuous daylight autonomy of 3rd daylight controlled gain,[%]
Airnode Outputs (Daylight):,431,DADLC3,Daylight autonomy of 3rd daylight controlled gain,[%]
Airnode Outputs (Daylight):,432,UDIDLC3,Useful daylight illuminance of 3rd daylight controlled gain,[%]
Airnode Outputs (Daylight):,435,STDFILE_DL,"Daylight standard output file is printed. The output file contains statistic outputs (continuous daylight autonomy, daylight autonomy and Useful daylight illuminance). For these variables the selected ON/OFF factor (schedule which is 0or 1) is used. In addition, the selected ON/OFF factor is set to this NType.",
Balance Outputs (printing standard files):,901,BAL_1,Solar Balance for Zones is printed,[kJ/hr]
Balance Outputs (printing standard files):,902,BAL_2,Solar Balance for Sum of all Zones is printed,[kJ/hr]
Balance Outputs (printing standard files):,903,BAL_3,Solar Balance for External Window is printed,[kJ/hr]
Balance Outputs (printing standard files):,904,BAL_4,Energy Balance for Zones is printed,[kJ/hr]
Balance Outputs (printing standard files):,905,BAL_5,Energy Balance for Sum of all Zonesis printed,[kJ/hr]
Balance Outputs (printing standard files):,906,BAL_6,Energy Balance for Wall Surfacesis printed,[kJ/hr]
Balance Outputs (printing standard files):,907,BAL_7,Moisture Balance for Airnodes is printed,[kJ/hr]
Balance Outputs (printing standard files):,908,BAL_8,Moisture Balance for Sum of all Airnodes is printed,[kJ/hr]
Zone Outputs corresponding to Balance 901:,910,B1_QBAL,BAL 1: solar balance for one zone,[kJ/hr]
Zone Outputs corresponding to Balance 901:,911,B1_QSEXT,BAL 1: total external solar radiation on all windows of one zone including frame,[kJ/hr]
Zone Outputs corresponding to Balance 901:,912,B1_QSADJ,BAL 1: solar gains due to exchange with adjacent zones (gains +; Losses -). Including multiple reflection.,[kJ/hr]
Zone Outputs corresponding to Balance 901:,913,B1_QBREFG,BAL 1: solar blocked due to reflection of glazing layers and shading devices of all windows of a zone,[kJ/hr]
Zone Outputs corresponding to Balance 901:,914,B1_QBABSG,BAL 1: solar blocked due to absorption on glazing layers and shading devices of all external windows (= secondary heat flux going out but without convective loss from gap ventilation (see B1_QSHVOUT),[kJ/hr]
Zone Outputs corresponding to Balance 901:,915,B1_QBFRM,BAL 1: solar blocked due to frames of all windows of a zone. (absorbed and reflected solar),[kJ/hr]
Zone Outputs corresponding to Balance 901:,916,B1_QBGSHD,solar blocked due to geometric shading of all external windows of a zone (Trnsys 17: solar blocked due to geometeric and external shading),[kJ/hr]
Zone Outputs corresponding to Balance 901:,917,B1_QSLOSS,BAL 1: solar radiation leaving zone by transmission through external windows of zone,[kJ/hr]
Zone Outputs corresponding to Balance 901:,918,B1_QSGWIN,BAL 1: absorbed solar gains on all windows of zones going inside (secondary heatflux of total window but without convective gain from gap ventilation (see B1_QISHCCI)),[kJ/hr]
Zone Outputs corresponding to Balance 901:,919,B1_QBRISHD,Obsolete (incl in B1_QBREFG for Trnsys 18) solar blocked due to reflection on internal shading devices of all windows of a zone,[kJ/hr]
Zone Outputs corresponding to Balance 901:,920,B1_QISHCCI,BAL 1: convective energy gain of zone due to absorbed solar going inside by ventilation of internal shading device,[kJ/hr]
Zone Outputs corresponding to Balance 901:,921,B1_QSGWALL,BAL 1: absorbed solar radiation on all walls of zone,[kJ/hr]
Zone Outputs corresponding to Balance 901:,961,B1_QSOLAIR,BAL 1: convective energy gain of zone due transmitted solar radiation through external windows which is transformed immediately into a con. heat flow to internal air (same as balance 4: NType 961),[kJ/hr]
Zone Outputs corresponding to Balance 901:,922,B1_QSHVOUT,BAL 1: convective loss due to absorbed solar going out by gap ventilation,[kJ/hr]
Surface Outputs corresponding to Balance 903:,930,B3_QBAL,BAL 3: solar balance for one external window should be always 0,[kJ/hr]
Surface Outputs corresponding to Balance 903:,931,B3_QSEXT,BAL 3: total external solar radiation on the external windows including frame,[kJ/hr]
Surface Outputs corresponding to Balance 903:,932,B3_QBGSHD,solar blocked due to geometric shading of external windows of a zone (Trnsys 17: solar blocked due to geometeric and external shading),[kJ/hr]
Surface Outputs corresponding to Balance 903:,933,B3_QBFRM,BAL 3: solar blocked due to frame of external window of a zone,[kJ/hr]
Surface Outputs corresponding to Balance 903:,934,B3_QBREFG,BAL 3: solar blocked due to reflection of glazing and shading layers of external window,[kJ/hr]
Surface Outputs corresponding to Balance 903:,935,B3_QBABSG,BAL 3: solar blocked due to absorption on glazing and shading devices of external windows (= secondary heat flux of external window going out without convective loss from gap ventilation (see B3_QSHVOUT),[kJ/hr]
Surface Outputs corresponding to Balance 903:,946,B3_ QSHVOUT,BAL3: solar radiation absorbed and removed by ventilated gaps to outside,
Surface Outputs corresponding to Balance 903:,936,B3_QBRISHD,obsolete (incl. in B3_QBREFG): solar blocked due to reflection on internal shading device (only shortwave radiation included,[kJ/hr]
Surface Outputs corresponding to Balance 903:,937,B3_QBLWISHD,obsolete (incl. in B3_QBABSG): solar blocked due to reflection on internal shading device (part which is absorbed and then going out only longwave),[kJ/hr]
Surface Outputs corresponding to Balance 903:,938,B3_QSHFPR,BAL 3: secondary heat flux of external window only primary solar no reflected radiation or radiation through other windows,[kJ/hr]
Surface Outputs corresponding to Balance 903:,939,B3_QSTRNS,BAL 3: short wave transmission through external window to zone (primary solar),[kJ/hr]
Surface Outputs corresponding to Balance 903:,940,gtot,total g-value (SHGC) of external window,[kJ/hr]
Surface Outputs corresponding to Balance 903:,,fc_Gshade,shading coefficient of geometric shading,[kJ/hr]
Surface Outputs corresponding to Balance 903:,941,fc_Eshade,shading coefficient of external shading (for complex fenestration model: not available),[kJ/hr]
Surface Outputs corresponding to Balance 903:,942,gframe,g-value (SHGC) of frame of external window,[kJ/hr]
Surface Outputs corresponding to Balance 903:,943,gglas,g-value (SHGC) of glas of external window,[kJ/hr]
Surface Outputs corresponding to Balance 903:,944,fc_Ishade,shading coefficient internal shading (for complex fenestration model: not available),[kJ/hr]
Surface Outputs corresponding to Balance 903:,945,AI,angle of incidence for window balance,[kJ/hr]
Zone Outputs corresponding to Balance 904:,950,B4_QBAL,BAL 4: energy balance for zone should be always close to 0,[kJ/hr]
Zone Outputs corresponding to Balance 904:,951,B4_DQAIRdt,BAL 4: change of internal energy of zone,[kJ/hr]
Zone Outputs corresponding to Balance 904:,952,B4_QHEAT,BAL 4: power of ideal heating (convective+radiative),[kJ/hr]
Zone Outputs corresponding to Balance 904:,953,B4_QCOOL,BAL 4: power of ideal cooling,[kJ/hr]
Zone Outputs corresponding to Balance 904:,954,B4_QINF,BAL 4: infiltration gains,[kJ/hr]
Zone Outputs corresponding to Balance 904:,955,B4_QVENT,BAL 4: ventilation gains,[kJ/hr]
Zone Outputs corresponding to Balance 904:,956,B4_QCOUP,BAL 4: coupling gains,[kJ/hr]
Zone Outputs corresponding to Balance 904:,957,B4_QTRANS,BAL 4: transmission into the surface from inner surface node,[kJ/hr]
Zone Outputs corresponding to Balance 904:,958,B4_QGINT,BAL 4: internal gains (convective+radiative),[kJ/hr]
Zone Outputs corresponding to Balance 904:,959,B4_QWGAIN,BAL 4: wall gains,[kJ/hr]
Zone Outputs corresponding to Balance 904:,960,B4_QSOL,BAL 4: absorbed solar gains on all inside surfaces of zone,[kJ/hr]
Zone Outputs corresponding to Balance 904:,961,B4_QSOLAIR,BAL 4: convective energy gain of zone due transmitted solar radiation through external windows which is transformed immediately into a con. heat flow to internal air,[kJ/hr]
Surface Outputs for walls corresponding to Balance 906:,972,B6_BAL,BAL 6: energy balance for a wall should be always 0,[kJ/hr]
Surface Outputs for walls corresponding to Balance 906:,973,B6_DQWALL,BAL 6 : change of internal energy of wall,[kJ/hr]
Surface Outputs for walls corresponding to Balance 906:,974,B6_QCOMI,BAL 6 : combined heat flux to inside (going into zone+; going into wall -),[kJ/hr]
Surface Outputs for walls corresponding to Balance 906:,975,B6_QCOMO,BAL 6 : combined heat flux to outside (going to outside-; going into wall +),[kJ/hr]
Surface Outputs for walls corresponding to Balance 906:,976,B6_QRADGI,BAL 6 : Total radiative gains for inner surface node,[kJ/hr]
Surface Outputs for walls corresponding to Balance 906:,977,B6_QRADGO,BAL 6 : Total radiative gains for outside surface node,[kJ/hr]
Surface Outputs for walls corresponding to Balance 906:,978,B6_QALG,BAL 6 : Total energy gains by an active layer or a chilled ceiling (heating -; cooling +).,[kJ/hr]
Airnode Outputs corresponding to Balance 907:,980,B7_MWBAL,BAL 7: moisture balance of an airnode should be always 0,[kg/hr]
Airnode Outputs corresponding to Balance 907:,981,B7_MDWAIR,BAL 7: change of water stored in the air of airnode,[kg/hr]
Airnode Outputs corresponding to Balance 907:,982,B7_DMWBUF,BAL 7: change of water stored in the surfaces of airnode for the detailed humidity,[kg/hr]
Airnode Outputs corresponding to Balance 907:,983,B7_MWINF,BAL 7: water gain of zone due to infiltration,[kg/hr]
Airnode Outputs corresponding to Balance 907:,984,B7_MWVENT,BAL 7: water gain of zone due to ventilation,[kg/hr]
Airnode Outputs corresponding to Balance 907:,985,B7_MWCOUP,BAL 7: water gain of zone due to coupling,[kg/hr]
Airnode Outputs corresponding to Balance 907:,986,B7_MWIGAIN,BAL 7: water gain from internal loads,[kg/hr]
Airnode Outputs corresponding to Balance 907:,987,B7_MWHUM,BAL 7: water gain due to ideal humidification of heating type,[kg/hr]
Airnode Outputs corresponding to Balance 907:,988,B7_MWDHUM,BAL 7: water loss due to ideal dehumidification of cooling type,[kg/hr]
"""


def _is_type56(type_id: str) -> bool:
    """Check if the type_id refers to Type56."""
    normalized = type_id.strip().lower()
    return normalized in ("type56", "56", "type 56")


def _normalize_type56_building_file_path(building_file_path: Optional[str]) -> Optional[str]:
    """Normalize Type56 building file paths for Docker-mounted TRNSYS data."""
    if not building_file_path:
        return building_file_path

    raw_path = str(building_file_path).strip().strip('"')
    if not raw_path:
        return raw_path

    normalized = raw_path.replace("\\", "/")
    if normalized.startswith("/mnt/trnsys/"):
        return normalized

    match = re.match(r"^[Dd]:/TRNSYS18_0[34]/(.+)$", normalized, re.IGNORECASE)
    if not match:
        return raw_path

    docker_path = f"/mnt/trnsys/TRNSYS18_04/{match.group(1)}"
    if os.path.exists(docker_path) or not os.path.exists(raw_path):
        return docker_path

    return raw_path


def _read_text_file(file_path: str) -> str:
    """Read text file with UTF-8 encoding."""
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _normalize_unit(unit: str) -> str:
    """Normalize unit string."""
    unit = (unit or "").strip()
    if not unit:
        return "any"
    m = re.match(r"^\[(.*)\]$", unit)
    if m:
        inner = m.group(1).strip()
        return inner if inner else "any"
    return unit


def _load_type56_output_units() -> Tuple[Dict[Tuple[int, str], str], Dict[str, str]]:
    """Load Type56 optional output units from CSV data."""
    by_pair: Dict[Tuple[int, str], str] = {}
    by_label: Dict[str, str] = {}

    reader = csv.DictReader(io.StringIO(_TRNSYS_TYPE56_OUTPUTS_CSV))
    for row in reader:
        ntype_raw = (row.get("NType#") or "").strip()
        label_raw = (row.get("Label") or "").strip()
        unit_raw = (row.get("Unit") or "").strip()
        if not label_raw:
            continue

        label_key = label_raw.upper()
        unit_value = _normalize_unit(unit_raw)

        if ntype_raw.isdigit():
            key = (int(ntype_raw), label_key)
            if key not in by_pair:
                by_pair[key] = unit_value

        if label_key not in by_label:
            by_label[label_key] = unit_value

    return by_pair, by_label


def _parse_building_file_inputs(building_file_path: str) -> List[Dict[str, str]]:
    """Parse inputs from .b18/.b17/.bui file."""
    content = _read_text_file(building_file_path)
    lines = content.splitlines()

    input_names: List[str] = []
    descriptions: Dict[str, Tuple[str, str]] = {}

    i = 0
    while i < len(lines):
        line = lines[i]
        m = _INPUTS_LINE_RE.match(line)
        if m:
            tail = (m.group(1) or "").strip()
            if tail:
                input_names = tail.split()
            i += 1
            break
        i += 1

    desc_header_found = False
    while i < len(lines):
        line = lines[i]
        if _INPUTS_DESC_HEADER_RE.match(line):
            desc_header_found = True
            i += 1
            break
        stripped = line.strip()
        if not stripped or line.lstrip().startswith("*"):
            i += 1
            continue
        break

    if desc_header_found:
        while i < len(lines):
            line = lines[i].rstrip("\n")
            if not line.strip():
                i += 1
                continue
            if line.lstrip().startswith("*"):
                break
            if re.match(r"^\s*[A-Z_]+\b.*=", line):
                break
            m = _INPUTS_DESC_LINE_RE.match(line)
            if m:
                name = m.group(1).strip()
                unit = m.group(2).strip() or "any"
                desc = m.group(3).strip()
                descriptions[name] = (unit, desc)
            i += 1

    extracted: List[Dict[str, str]] = []
    for name in input_names:
        unit, desc = descriptions.get(name, ("any", ""))
        extracted.append({"name": name, "unit": unit, "description": desc})
    return extracted


def _parse_building_file_outputs(building_file_path: str) -> List[Dict[str, Any]]:
    """Parse outputs from .b18/.b17/.bui file."""
    content = _read_text_file(building_file_path)
    lines = content.splitlines()

    def _split_airnodes(raw: str) -> List[str]:
        return [p for p in re.split(r"\s+", (raw or "").strip()) if p]

    def _parse_surfaces(raw: str) -> List[int]:
        surfaces: List[int] = []
        for token in (raw or "").split(","):
            token = token.strip()
            if token.isdigit():
                surfaces.append(int(token))
        return surfaces

    i = 0
    while i < len(lines):
        if _OUTPUTS_HEADER_RE.match(lines[i]):
            i += 1
            break
        i += 1

    current_airnodes: List[str] = []
    extracted: List[Dict[str, Any]] = []

    while i < len(lines):
        line = lines[i].rstrip("\n")
        if line.lstrip().startswith("*"):
            break

        m_air = _AIRNODES_RE.match(line)
        if m_air:
            current_airnodes = _split_airnodes(m_air.group(1))
            i += 1
            continue

        m_types = _NTYPES_RE.match(line)
        if m_types:
            candidate = m_types.group(1).strip()
            m_surf = _OUTPUT_SURF_ENTRY_RE.match(candidate)
            m_entry = _OUTPUT_TYPE_ENTRY_RE.match(candidate)
            if current_airnodes and m_surf:
                ntype = int(m_surf.group(1))
                surfaces = _parse_surfaces(m_surf.group(2))
                label = m_surf.group(3).strip()
                desc = m_surf.group(4).strip()
                for airnode in current_airnodes:
                    for surf in surfaces:
                        extracted.append({"ntype": ntype, "label": label, "desc": desc, "airnode": airnode, "surf": surf})
            elif current_airnodes and m_entry:
                ntype = int(m_entry.group(1))
                label = m_entry.group(2).strip()
                desc = m_entry.group(3).strip()
                for airnode in current_airnodes:
                    extracted.append({"ntype": ntype, "label": label, "desc": desc, "airnode": airnode})
            i += 1
            continue

        m_cont = _NTYPES_CONT_RE.match(line)
        if m_cont:
            candidate = m_cont.group(1).strip()
            m_surf = _OUTPUT_SURF_ENTRY_RE.match(candidate)
            m_entry = _OUTPUT_TYPE_ENTRY_RE.match(candidate)
            if current_airnodes and m_surf:
                ntype = int(m_surf.group(1))
                surfaces = _parse_surfaces(m_surf.group(2))
                label = m_surf.group(3).strip()
                desc = m_surf.group(4).strip()
                for airnode in current_airnodes:
                    for surf in surfaces:
                        extracted.append({"ntype": ntype, "label": label, "desc": desc, "airnode": airnode, "surf": surf})
            elif current_airnodes and m_entry:
                ntype = int(m_entry.group(1))
                label = m_entry.group(2).strip()
                desc = m_entry.group(3).strip()
                for airnode in current_airnodes:
                    extracted.append({"ntype": ntype, "label": label, "desc": desc, "airnode": airnode})
            i += 1
            continue

        i += 1

    return extracted


def _get_type56_spec_from_building_file(building_file_path: str) -> Dict[str, Any]:
    """
    Get Type56 specification with inputs/outputs from .b18/.b17/.bui file.

    Args:
        building_file_path: Absolute path to .b18/.b17/.bui file

    Returns:
        Type56 TMF specification with building-specific inputs/outputs
    """
    if not os.path.isabs(building_file_path):
        raise ValueError("building_file_path must be an absolute path to a .b18/.b17/.bui file")
    if not os.path.exists(building_file_path):
        raise FileNotFoundError(f"File not found: {building_file_path}")

    ext = os.path.splitext(building_file_path)[1].lower()
    if ext not in {".b18", ".b17", ".bui"}:
        raise ValueError("Only .b18/.b17/.bui files are supported")

    template = deepcopy(_TYPE56_TEMPLATE["Type56.tmf"])
    template["external_files"] = [{"prompt": "Building description file", "default_path": building_file_path}]

    existing_inputs = template.get("inputs", [])
    extracted_inputs = _parse_building_file_inputs(building_file_path)
    extracted_outputs = _parse_building_file_outputs(building_file_path)
    output_units_by_pair, output_units_by_label = _load_type56_output_units()

    # Add extracted inputs
    next_index = len(existing_inputs) + 1
    new_inputs: List[Dict[str, Any]] = []
    for item in extracted_inputs:
        new_inputs.append({
            "index": next_index,
            "name": item["name"],
            "description": item["description"],
            "unit": item["unit"],
            "default_value": "0",
        })
        next_index += 1

    template["inputs"] = existing_inputs + new_inputs

    # Build outputs
    outputs: List[Dict[str, Any]] = []
    out_index = 1
    seen_output_names: set = set()
    for item in extracted_outputs:
        label = str(item["label"]).strip()
        ntype = int(item["ntype"])
        unit = output_units_by_pair.get((ntype, label.upper()))
        if not unit:
            unit = output_units_by_label.get(label.upper())
        if not unit:
            unit = "any"

        airnode_raw = str(item.get("airnode") or "")
        airnode_for_name = re.sub(r"\s+", "_", airnode_raw).strip("_")
        surf = item.get("surf")

        if isinstance(surf, int):
            name_body = f"{label}_S{surf}_{airnode_for_name}" if airnode_for_name else f"{label}_S{surf}"
            description = f"{item['desc']}, S{surf}, {airnode_raw}"
        else:
            name_body = f"{label}_{airnode_for_name}"
            description = f"{item['desc']}, {airnode_raw}"

        seen_output_names.add(name_body)
        outputs.append({
            "index": out_index,
            "name": name_body,
            "description": description,
            "unit": unit,
            "default_value": "0",
        })
        out_index += 1

    template["outputs"] = outputs if outputs else [{"index": 1, "name": "Placeholder", "description": "No outputs defined", "unit": "any"}]

    # Update variable counts
    template["variable_counts"]["inputs"] = len(template["inputs"])
    template["variable_counts"]["outputs"] = len(template["outputs"])

    return template


def _get_type56_base_spec() -> Dict[str, Any]:
    """Get Type56 base specification without building file."""
    return deepcopy(_TYPE56_TEMPLATE["Type56.tmf"])


def _get_tmf_data() -> Dict[str, Any]:
    """
    Load and cache TMF data from the parsed JSON file.

    Returns:
        Dictionary containing all TMF component specifications.
    """
    global _tmf_data_cache

    if _tmf_data_cache is not None:
        return _tmf_data_cache

    try:
        from ...core.config import get_settings
        settings = get_settings()
        tmf_path = settings.tmf_parsed_path

        if not tmf_path.exists():
            logger.error(f"TMF parsed file not found: {tmf_path}")
            return {"results": {}}

        with open(tmf_path, "r", encoding="utf-8") as f:
            _tmf_data_cache = json.load(f)

        logger.info(f"Loaded TMF data: {_tmf_data_cache.get('total_tmf_files', 0)} components")
        return _tmf_data_cache

    except Exception as e:
        logger.error(f"Error loading TMF data: {e}")
        return {"results": {}}


def _normalize_type_id(type_id: str) -> str:
    """
    Normalize type ID to standard format for TMF lookup.

    CRITICAL: The TMF filename without ".tmf" is the UNIQUE component ID.
    This function first checks if the input directly matches a TMF filename.
    Only applies "Type" prefix normalization as a fallback for numeric-only inputs.

    The 765 TMF filenames include non-standard names like:
    - "19Wind-1", "APart", "Pacemaker", "TRNOPT" (no "Type" prefix)
    - "TYPE54c", "TYPE581a" (uppercase "TYPE" prefix)
    - "Type15-6", "Type65d" (standard "Type" prefix)

    Examples:
        "Type15-6" -> "Type15-6"
        "type15-6" -> "Type15-6" (case normalization for standard Type prefix)
        "TYPE54c" -> "TYPE54c" (preserved as-is, matches TMF filename)
        "19Wind-1" -> "19Wind-1" (preserved as-is, matches TMF filename)
        "APart" -> "APart" (preserved as-is, matches TMF filename)
        "15-6" -> "Type15-6" (numeric input, add Type prefix)
        "927" -> "Type927" (numeric input, add Type prefix)
    """
    type_id = type_id.strip()

    # If input starts with a digit, it's likely a numeric-only ID -> add "Type" prefix
    if type_id and type_id[0].isdigit():
        return f"Type{type_id}"

    # If starts with "type" (lowercase) but not "TYPE" (uppercase), normalize to "Type"
    if type_id.lower().startswith("type") and not type_id.startswith("TYPE"):
        suffix = type_id[4:]  # Everything after "type" or "Type"
        return f"Type{suffix}"

    # Otherwise, preserve as-is (handles "TYPE54c", "APart", "19Wind-1", "Pacemaker", etc.)
    return type_id


def _find_tmf_by_type(type_id: str, building_file_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Find TMF entries matching the given type ID.

    CRITICAL: The TMF filename (without .tmf) is the UNIQUE component ID.
    This function prioritizes EXACT matches against the 765 TMF filenames.

    Lookup order:
    1. Direct exact match: type_id + ".tmf" in TMF registry
    2. Case-insensitive match against all 765 TMF filenames
    3. Normalized match (add "Type" prefix for numeric inputs)
    4. Variant search (fallback for imprecise IDs like "Type15")

    Special handling for Type56:
    - If building_file_path is provides, returns full specification with building-specific I/O
    - If building_file_path is not provides, returns base Type56 template

    Args:
        type_id: Component ID to search for. Should be the exact TMF filename without .tmf:
            - "Type15-6" (exact match for Type15-6.tmf)
            - "Type65d" (exact match for Type65d.tmf)
            - "TYPE54c" (exact match for TYPE54c.tmf)
            - "19Wind-1" (exact match for 19Wind-1.tmf)
            - "Pacemaker" (exact match for Pacemaker.tmf)
        building_file_path: Optional path to .b18/.b17/.bui file for Type56

    Returns:
        List of matching TMF entries. For precise IDs, returns exactly one match.
        For imprecise IDs (like "Type15"), returns all variants as a warning.
    """
    # Special handling for Type56
    if _is_type56(type_id):
        if building_file_path and os.path.exists(building_file_path):
            try:
                type56_spec = _get_type56_spec_from_building_file(building_file_path)
                return [type56_spec]
            except Exception as e:
                logger.warning(f"Failed to parse Type56 from {building_file_path}: {e}, using base template")
                return [_get_type56_base_spec()]
        else:
            return [_get_type56_base_spec()]

    tmf_data = _get_tmf_data()
    results = tmf_data.get("results", {})

    # Step 1: Try DIRECT exact match first (type_id as-is + ".tmf")
    direct_key = f"{type_id.strip()}.tmf"
    if direct_key in results:
        tmf_info = results[direct_key]
        comp_id = direct_key.replace(".tmf", "")
        return [{"tmf_name": direct_key, "component_id": comp_id, **tmf_info}]

    # Step 2: Try case-insensitive match against all TMF filenames
    type_id_lower = type_id.strip().lower()
    for tmf_name in results:
        if tmf_name.lower() == f"{type_id_lower}.tmf":
            tmf_info = results[tmf_name]
            comp_id = tmf_name.replace(".tmf", "")
            return [{"tmf_name": tmf_name, "component_id": comp_id, **tmf_info}]

    # Step 3: Try normalized match (handles "15-6" -> "Type15-6", "type15-6" -> "Type15-6")
    normalized_id = _normalize_type_id(type_id)
    tmf_key = f"{normalized_id}.tmf"
    if tmf_key in results:
        tmf_info = results[tmf_key]
        return [{"tmf_name": tmf_key, "component_id": normalized_id, **tmf_info}]

    # Step 4: Fallback - search for variants (warn that this is imprecise)
    matches = []
    base_type = normalized_id.rstrip('abcdefghijklmnopqrstuvwxyz').split('-')[0]

    for tmf_name, tmf_info in results.items():
        tmf_base = tmf_name.replace('.tmf', '').rstrip('abcdefghijklmnopqrstuvwxyz').split('-')[0]
        if tmf_base == base_type:
            component_id = tmf_name.replace('.tmf', '')
            matches.append({"tmf_name": tmf_name, "component_id": component_id, **tmf_info})

    return matches


def _format_parameter(param: Dict[str, Any]) -> str:
    """Format a single parameter for display (English output)."""
    name = param.get("name", "Unknown")
    desc = param.get("description", "")[:100]
    default = param.get("default_value", "N/A")
    unit = param.get("unit", "")
    min_val = param.get("min_value", "")
    max_val = param.get("max_value", "")

    result = f"  {param.get('index', '?')}. {name}"
    if default:
        result += f" (default: {default})"
    if unit:
        result += f" [{unit}]"
    if min_val or max_val:
        result += f" range: [{min_val}, {max_val}]"
    if desc:
        result += f"\n     {desc}"

    return result


def _format_io(io_item: Dict[str, Any], io_type: str) -> str:
    """Format a single input/output for display (English output)."""
    name = io_item.get("name", "Unknown")
    desc = io_item.get("description", "")[:80]
    unit = io_item.get("unit", "")
    default = io_item.get("default_value", "")

    result = f"  {io_item.get('index', '?')}. {name}"
    if unit:
        result += f" [{unit}]"
    if default and io_type == "input":
        result += f" (initial: {default})"
    if desc:
        result += f"\n     {desc}"

    return result


def _get_type15_notes(component_id: str) -> str:
    """
    Get special notes for Type15 weather components regarding parameter expansion.

    Args:
        component_id: Component ID (e.g., "Type15-2", "Type15-6")

    Returns:
        Notes string if component is a supported Type15 variant, empty string otherwise.
    """
    # Supported Type15 variants that require parameter expansion
    type15_variants = [
        "Type15-1", "Type15-2", "Type15-3", "Type15-4",
        "Type15-6", "Type15-7", "Type15-10", "Type15-TMY3"
    ]

    if component_id not in type15_variants:
        return ""

    notes = """
CRITICAL NOTES for Type15 Weather Components - Parameter Expansion Required:

1. PARAMETER COUNT FORMULA:
   TOTAL_PARAMETERS = 6 (BASE_PARAMETERS) + 3 × (Number of surfaces)

   Examples:
   - Number of surfaces = 1 → 9 parameters (6 + 3×1)
   - Number of surfaces = 5 → 21 parameters (6 + 3×5)
   - Number of surfaces = 7 → 27 parameters (6 + 3×7)

2. BASE PARAMETERS (indices 1-6):
   - Parameter 1: Data file logical unit
   - Parameter 2: Data file format
   - Parameter 3: Tilted Surface Radiation Mode
   - Parameter 4: Ground reflectance - no snow
   - Parameter 5: Ground reflectance - snow cover
   - Parameter 6: Number of surfaces

3. SURFACE PARAMETERS (3 parameters per surface):
   For each surface i (where i = 1 to Number of surfaces):
   - Parameter (6 + 3×(i-1) + 1): Tracking mode for surface i
   - Parameter (6 + 3×(i-1) + 2): Slope of surface i [degrees]
   - Parameter (6 + 3×(i-1) + 3): Azimuth of surface i [degrees]

4. RECOMMENDED CONFIGURATION:
   Set "Number of surfaces" = 5 (resulting in 21 total parameters)
   This provides good coverage for typical building orientations:
   - Surface 1: Horizontal (roof/ground) - Tracking=1, Slope=0, Azimuth=0
   - Surface 2: South-facing vertical - Tracking=1, Slope=90, Azimuth=0
   - Surface 3: West-facing vertical - Tracking=1, Slope=90, Azimuth=90
   - Surface 4: North-facing vertical - Tracking=1, Slope=90, Azimuth=180
   - Surface 5: East-facing vertical - Tracking=1, Slope=90, Azimuth=270

5. IMPORTANT:
   - The TMF file shows only 9 parameters (base + 1 surface example)
   - You MUST expand parameters based on the actual "Number of surfaces" value
   - DO NOT use the TMF parameter count (9) directly for validation
   - Always calculate: 6 + 3 × (Number of surfaces)
"""
    return notes.strip()


# === TMF Tools ===

@tool
def get_tmf_spec(type_id: str, building_file_path: Optional[str] = None) -> str:
    """
    Get TMF specification for a TRNSYS component.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
    The 765 TMF filenames (XXX.tmf -> XXX) are the ONLY valid component IDs.
    Do NOT use type_number as component ID - type_number is NOT unique!

    Special handling for Type56 (Multi-Zone Building):
    - Type56 requires external .b18/.b17/.bui file to get complete I/O info
    - If building_file_path is provides, returns full specification with building-specific inputs/outputs
    - If building_file_path is not provides, returns base Type56 template with 3 parameters and 7 base inputs

    Args:
        type_id: Exact TMF filename without .tmf (the unique component ID). Examples:
            - "Type15-6" (from Type15-6.tmf) - Weather data (Meteonorm format)
            - "Type15-2" (from Type15-2.tmf) - Weather data (TMY2 format)
            - "Type65d" (from Type65d.tmf) - Online plotter (no file output)
            - "Type65a" (from Type65a.tmf) - Online plotter (with file output)
            - "Type56" (from Type56.tmf) - Multi-zone building
            - "TYPE54c" (from TYPE54c.tmf) - Weather generator
            - "19Wind-1" (from 19Wind-1.tmf) - Window for Type 19
            - "Pacemaker" (from Pacemaker.tmf) - Real time pacer
        building_file_path: Optional absolute path to .b18/.b17/.bui file for Type56.
            Only used when type_id is Type56.

    Returns:
        Formatted TMF specification including parameters, inputs, outputs.
        IMPORTANT: Includes model_path field - use this EXACT path for *$MODEL in DCK!
        If multiple variants found, lists all with a warning to use precise ID.

    Example:
        >>> get_tmf_spec("Type15-6")
        "【Type15-6 - Weather Data Processor】\nmodel_path: C:\\TRNSYS18\\...\\Type15-6.tmf\n..."
        >>> get_tmf_spec("Type56", building_file_path="D:\\project\\building.b18")
        "【Type56 - Multi-Zone Building】\n... (with building-specific inputs/outputs)"
    """
    try:
        building_file_path = _normalize_type56_building_file_path(building_file_path)
        matches = _find_tmf_by_type(type_id, building_file_path=building_file_path)

        if not matches:
            return f"ERROR: No TMF specification found for '{type_id}'. Please check the component ID."

        # Format results
        if len(matches) == 1:
            parts = [f"Found exact match for component ID: {matches[0].get('component_id', type_id)}\n"]
        else:
            parts = [f"WARNING: Found {len(matches)} variants. Please use a PRECISE component ID!\n"]
            parts.append("Available variants:")
            for m in matches:
                comp_id = m.get('component_id', 'Unknown')
                params = m.get('variable_counts', {}).get('parameters', 0)
                inputs = m.get('variable_counts', {}).get('inputs', 0)
                parts.append(f"  - {comp_id}: {params} parameters, {inputs} inputs")
            parts.append("\n")

        for i, tmf in enumerate(matches[:3], 1):  # Limit to 3 variants
            tmf_name = tmf.get("tmf_name", "Unknown")
            component_id = tmf.get("component_id", "Unknown")
            model_name = tmf.get("model_name", "Unknown")
            type_num = tmf.get("type_number", "?")
            description = tmf.get("description", "")[:300]
            tmf_path = tmf.get("tmf_path", "")  # Full path to TMF file

            # Convert absolute path to relative path (replace everything before and including Proformas\ with .\)
            # Example: d:\TRNSYS18_03\Studio\Proformas\Weather\Type15-6.tmf -> .\Weather\Type15-6.tmf
            if tmf_path:
                # Find "Proformas\" or "Proformas/" in the path (case insensitive)
                import re as re_module
                proformas_match = re_module.search(r'[Pp]roformas[\\\/]', tmf_path)
                if proformas_match:
                    # Get everything after "Proformas\"
                    relative_path = ".\\" + tmf_path[proformas_match.end():]
                    tmf_path = relative_path

            var_counts = tmf.get("variable_counts", {})
            param_count = var_counts.get("parameters", 0)
            input_count = var_counts.get("inputs", 0)
            output_count = var_counts.get("outputs", 0)

            parts.append(f"\n{'='*60}")
            parts.append(f"【{i}. {component_id} - {model_name}】")
            parts.append(f"TMF File: {tmf_name}")
            # CRITICAL: Include model_path for DCK generation
            parts.append(f"model_path: {tmf_path}")
            parts.append(f"Description: {description}")
            parts.append(f"\nVariable Counts:")
            parts.append(f"  - PARAMETERS: {param_count}")
            parts.append(f"  - INPUTS: {input_count}")
            parts.append(f"  - OUTPUTS: {output_count}")

            # Parameters
            parameters = tmf.get("parameters", [])
            if parameters:
                parts.append(f"\n【PARAMETERS】({len(parameters)} total)")
                for param in parameters[:15]:  # Show more parameters
                    parts.append(_format_parameter(param))
                if len(parameters) > 15:
                    parts.append(f"  ... and {len(parameters) - 15} more parameters")

            # Inputs
            inputs = tmf.get("inputs", [])
            if inputs:
                parts.append(f"\n【INPUTS】({len(inputs)} total)")
                for inp in inputs[:10]:
                    parts.append(_format_io(inp, "input"))
                if len(inputs) > 10:
                    parts.append(f"  ... and {len(inputs) - 10} more inputs")

            # Outputs
            outputs = tmf.get("outputs", [])
            if outputs:
                parts.append(f"\n【OUTPUTS】({len(outputs)} total)")
                for out in outputs[:10]:
                    parts.append(_format_io(out, "output"))
                if len(outputs) > 10:
                    parts.append(f"  ... and {len(outputs) - 10} more outputs")

            # Specific properties (e.g., LABELS for plotters, unit strings)
            spec_props = tmf.get("specific_properties", [])
            if spec_props:
                parts.append(f"\n【Specific Properties】({len(spec_props)} total)")
                for sp in spec_props:
                    if isinstance(sp, dict):
                        sp_name = sp.get("specific_property", "Unknown")
                        sp_note = sp.get("note", "")
                        sp_default = sp.get("default", "")
                        line = f"  - {sp_name}"
                        if sp_note:
                            line += f" [note: {sp_note}]"
                        if sp_default:
                            line += f" (default: {sp_default})"
                        parts.append(line)

            # External files
            ext_files = tmf.get("external_files", [])
            if ext_files:
                parts.append(f"\n【External Files Required】")
                for ef in ext_files:
                    parts.append(f"  - {ef.get('prompt', 'Unknown')}")
                    parts.append(f"    Default path: {ef.get('default_path', 'N/A')}")

            # Add special notes for Type15 components
            notes = _get_type15_notes(component_id)
            if notes:
                parts.append(f"\n{'='*60}")
                parts.append("【SPECIAL NOTES】")
                parts.append(notes)

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in get_tmf_spec: {e}")
        return f"Error getting TMF specification: {str(e)}"


@tool
def validate_parameter_count(type_id: str, param_count: int, building_file_path: Optional[str] = None) -> str:
    """
    Validate component parameter count against TMF specification.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
    Different variants have different parameter counts!

    Note: For Type56, parameter count is always 3 regardless of building file.

    Args:
        type_id: Exact TMF filename without .tmf (e.g., "Type15-6", "Type65d", "TYPE54c")
        param_count: Number of parameters to validate
        building_file_path: Optional path to .b18/.b17/.bui file for Type56 (not used for parameter validation)

    Returns:
        Validation result with expected count and suggestions if mismatch.

    Example:
        >>> validate_parameter_count("Type15-6", 9)
        "OK: Type15-6 requires 9 parameters, provides 9."
    """
    try:
        building_file_path = _normalize_type56_building_file_path(building_file_path)
        matches = _find_tmf_by_type(type_id, building_file_path=building_file_path)

        if not matches:
            return f"WARNING: No TMF specification found for '{type_id}'. Cannot validate parameter count."

        # Check all variants
        results = []
        for tmf in matches:
            tmf_name = tmf.get("tmf_name", "Unknown")
            component_id = tmf.get("component_id", "Unknown")
            var_counts = tmf.get("variable_counts", {})
            expected_count = var_counts.get("parameters", 0)

            if param_count == expected_count:
                results.append(
                    f"OK: {component_id} requires {expected_count} parameters, "
                    f"provides {param_count}."
                )
            else:
                results.append(
                    f"ERROR: {component_id} requires {expected_count} parameters, "
                    f"but provides {param_count}.\n"
                    f"   Suggestion: Adjust PARAMETERS block to have exactly {expected_count} parameters."
                )

        # Add special notes for Type15 components
        result_text = "\n".join(results)

        # Check if any match is a Type15 variant and add notes
        for tmf in matches:
            component_id = tmf.get("component_id", "")
            notes = _get_type15_notes(component_id)
            if notes:
                result_text += f"\n\n{'='*60}\n"
                result_text += "【SPECIAL NOTES】\n"
                result_text += notes
                break  # Only add notes once even if multiple matches

        return result_text

    except Exception as e:
        logger.error(f"Error in validate_parameter_count: {e}")
        return f"Error validating parameter count: {str(e)}"


@tool
def validate_input_count(type_id: str, input_count: int, building_file_path: Optional[str] = None) -> str:
    """
    Validate component input count against TMF specification.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.

    NOTE: Some components have FLEXIBLE input counts determined by parameters or external files:
    - Type65a/b/c/d: Input count = Nb. of left-axis variables + Nb. of right-axis variables
    - Type56: Input count depends on building file (.b18/.b17/.bui) configuration
    - For these components, the TMF shows minimum inputs, actual count depends on configuration.

    Args:
        type_id: Exact TMF filename without .tmf (e.g., "Type15-6", "Type65d", "TYPE54c")
        input_count: Number of inputs to validate
        building_file_path: Optional path to .b18/.b17/.bui file for Type56

    Returns:
        Validation result with expected count and suggestions if mismatch.

    Example:
        >>> validate_input_count("Type927", 7)
        "OK: Type927 requires 7 inputs, provides 7."
    """
    try:
        building_file_path = _normalize_type56_building_file_path(building_file_path)
        matches = _find_tmf_by_type(type_id, building_file_path=building_file_path)

        if not matches:
            return f"WARNING: No TMF specification found for '{type_id}'. Cannot validate input count."

        # Check all variants
        results = []
        for tmf in matches:
            tmf_name = tmf.get("tmf_name", "Unknown")
            component_id = tmf.get("component_id", "Unknown")
            var_counts = tmf.get("variable_counts", {})
            expected_count = var_counts.get("inputs", 0)

            # Check for flexible input components (Category C in connection_generator)
            is_flexible = component_id in FLEXIBLE_INPUT_TYPES

            if is_flexible:
                results.append(
                    f"INFO: {component_id} has FLEXIBLE input count.\n"
                    f"   TMF shows {expected_count} base inputs.\n"
                    f"   Actual input count depends on controlling parameter values.\n"
                    f"   Provided {input_count} inputs - verify this matches your parameter settings."
                )
            elif input_count == expected_count:
                results.append(
                    f"OK: {component_id} requires {expected_count} inputs, "
                    f"provides {input_count}."
                )
            else:
                results.append(
                    f"ERROR: {component_id} requires {expected_count} inputs, "
                    f"but provides {input_count}.\n"
                    f"   Suggestion: Adjust INPUTS block to have exactly {expected_count} inputs."
                )

        return "\n".join(results)

    except Exception as e:
        logger.error(f"Error in validate_input_count: {e}")
        return f"Error validating input count: {str(e)}"


@tool
def get_component_io_mapping(type_id: str, building_file_path: Optional[str] = None) -> str:
    """
    Get input/output mapping for a TRNSYS component.

    CRITICAL: type_id MUST be the exact TMF filename without .tmf extension.
    This is essential for generating correct INPUTS blocks and connections.

    Special handling for Type56 (Multi-Zone Building):
    - If building_file_path is provides, returns complete I/O mapping from building file
    - If building_file_path is not provides, returns base Type56 I/O (7 base inputs)

    Args:
        type_id: Exact TMF filename without .tmf (e.g., "Type15-6", "Type65d", "TYPE54c")
        building_file_path: Optional absolute path to .b18/.b17/.bui file for Type56

    Returns:
        Formatted input/output mapping with port numbers, names, and units.

    Example:
        >>> get_component_io_mapping("Type927")
        "【Type927 Input/Output Mapping】\nInputs:\n  1. Chilled water inlet temp [C]\n..."
        >>> get_component_io_mapping("Type56", building_file_path="D:\\project\\building.b18")
        "【Type56 Input/Output Mapping】\n... (with building-specific I/O)"
    """
    try:
        building_file_path = _normalize_type56_building_file_path(building_file_path)
        matches = _find_tmf_by_type(type_id, building_file_path=building_file_path)

        if not matches:
            return f"ERROR: No I/O mapping found for '{type_id}'. Please check the component ID."

        parts = []

        for tmf in matches[:2]:  # Limit to 2 variants
            tmf_name = tmf.get("tmf_name", "Unknown")
            component_id = tmf.get("component_id", "Unknown")
            model_name = tmf.get("model_name", "Unknown")
            type_num = tmf.get("type_number", "?")

            parts.append(f"\n{'='*50}")
            parts.append(f"【{component_id} - {model_name} I/O Mapping】")
            parts.append(f"TMF: {tmf_name}")

            # Inputs
            inputs = tmf.get("inputs", [])
            if inputs:
                parts.append(f"\n【INPUT Ports】({len(inputs)} total)")
                parts.append("Format: [source_unit, source_output] or constant value")
                for inp in inputs:
                    idx = inp.get("index", "?")
                    name = inp.get("name", "Unknown")
                    unit = inp.get("unit", "")
                    default = inp.get("default_value", "0")
                    unit_str = f" [{unit}]" if unit else ""
                    parts.append(f"  Input {idx}: {name}{unit_str} (initial: {default})")
            else:
                parts.append("\n【INPUT Ports】No inputs")

            # Check for flexible inputs (Category C in connection_generator)
            is_flexible = component_id in FLEXIBLE_INPUT_TYPES
            if is_flexible:
                parts.append(f"\n  NOTE: {component_id} has FLEXIBLE input count!")
                parts.append("  Actual input count depends on controlling parameter values.")
                parts.append(
                    "  [!] CRITICAL — The input list above is only the SINGLE-INSTANCE base "
                    "template (one instance of whatever the controlling parameter counts — a "
                    "zone, tank/air node, inlet/outlet port, comparator, floor, control, "
                    "heating/cooling stage, load, level, etc.; NOT necessarily a zone). For a "
                    "MULTI-instance system, DO NOT assume the real port order is this template "
                    "repeated per instance (grouped/interleaved). Different flexible components "
                    "order their expanded ports DIFFERENTLY: some are BLOCKED by physical "
                    "quantity (e.g. Type152/Type151 list ALL zone temperatures first, THEN all "
                    "humidities, THEN all loads), others are GROUPED by instance (e.g. Type1255 "
                    "per-zone, Type649 per-inlet — all quantities of instance-1, then "
                    "instance-2). You CANNOT tell which order "
                    "from this base template alone. To get the real expanded port order, call "
                    "get_connection_pattern(source_type, '" + component_id + "', dck_filename=...) "
                    "on a reference DCK to learn the ordering MODE (blocked vs grouped), then "
                    "re-generate the ports for THIS task's actual instance count N following "
                    "that same mode — never blindly copy a different-N reference port-by-port."
                )

            # Outputs
            outputs = tmf.get("outputs", [])
            if outputs:
                parts.append(f"\n【OUTPUT Ports】({len(outputs)} total)")
                parts.append("Format: Can be referenced as [this_unit, output_number]")
                for out in outputs:
                    idx = out.get("index", "?")
                    name = out.get("name", "Unknown")
                    unit = out.get("unit", "")
                    unit_str = f" [{unit}]" if unit else ""
                    parts.append(f"  Output {idx}: {name}{unit_str}")
            else:
                parts.append("\n【OUTPUT Ports】No outputs")

            # Connection hints
            parts.append("\n【Connection Format】")
            parts.append("- In INPUTS block, each line corresponds to one input port")
            parts.append("- Format: source_unit,source_output  ! comment")
            parts.append("- Use 0,0 for unconnected inputs (uses initial value)")

        return "\n".join(parts)

    except Exception as e:
        logger.error(f"Error in get_component_io_mapping: {e}")
        return f"Error getting I/O mapping: {str(e)}"


# === Tool Collection Functions ===

def get_tmf_tools() -> list:
    """
    Get all TMF tools (4).

    Returns:
        List of @tool decorated functions.
    """
    return [
        get_tmf_spec,
        validate_parameter_count,
        validate_input_count,
        get_component_io_mapping,
    ]


def get_tmf_tools_for_agent(agent_role: str) -> list:
    """
    Get TMF tools for a specific agent role.

    Args:
        agent_role: Agent role name

    Returns:
        List of TMF tools appropriate for that role
    """
    role_tools = {
        "planner": [],  # Planner does not need TMF tools
        "component_selector": [get_tmf_spec],
        "parameter_configurator": [get_tmf_spec, validate_parameter_count],
        "connection_generator": [get_tmf_spec, get_component_io_mapping, validate_input_count],
        "equation_generator": [get_tmf_spec],  # Equation generator needs get_tmf_spec to look up port names for variable references and comments
        "layout_designer": [],  # Layout designer does not need TMF tools
        "validator": [get_tmf_spec, validate_parameter_count, validate_input_count],
        "supervisor": get_tmf_tools(),  # Supervisor can access all tools
    }

    return role_tools.get(agent_role, [])


# === Utility Functions ===

def clear_tmf_cache() -> None:
    """Clear the TMF data cache."""
    global _tmf_data_cache
    _tmf_data_cache = None
    logger.info("TMF cache cleared")


def get_tmf_stats() -> Dict[str, Any]:
    """
    Get statistics about loaded TMF data.

    Returns:
        Dictionary with TMF statistics.
    """
    tmf_data = _get_tmf_data()

    return {
        "total_components": tmf_data.get("total_tmf_files", 0),
        "total_parameters": tmf_data.get("total_variables", {}).get("parameters", 0),
        "total_inputs": tmf_data.get("total_variables", {}).get("inputs", 0),
        "total_outputs": tmf_data.get("total_variables", {}).get("outputs", 0),
        "cache_loaded": _tmf_data_cache is not None,
    }


def search_tmf_by_name(keyword: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Search TMF components by name or description.

    CRITICAL: Returns component_id (TMF filename without .tmf) as the unique identifier.
    The component_id is the ONLY valid way to reference a component.

    Args:
        keyword: Search keyword
        limit: Maximum results to return

    Returns:
        List of matching TMF entries with component_id as primary identifier.
    """
    tmf_data = _get_tmf_data()
    results = tmf_data.get("results", {})

    matches = []
    keyword_lower = keyword.lower()

    for tmf_name, tmf_info in results.items():
        model_name = tmf_info.get("model_name", "").lower()
        description = tmf_info.get("description", "").lower()

        if keyword_lower in model_name or keyword_lower in description or keyword_lower in tmf_name.lower():
            component_id = tmf_name.replace(".tmf", "")
            matches.append({
                "component_id": component_id,
                "tmf_name": tmf_name,
                "type_number": tmf_info.get("type_number"),
                "model_name": tmf_info.get("model_name"),
                "param_count": tmf_info.get("variable_counts", {}).get("parameters", 0),
                "input_count": tmf_info.get("variable_counts", {}).get("inputs", 0),
                "output_count": tmf_info.get("variable_counts", {}).get("outputs", 0),
            })

            if len(matches) >= limit:
                break

    return matches
