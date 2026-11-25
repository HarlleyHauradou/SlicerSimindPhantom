#-----------------------------------------------------------------------------
set(MODULE_NAME SimindPhantom)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  SimindPhantomLib/__init__.py
  SimindPhantomLib/core.py
  SimindPhantomLib/pipeline.py
  SimindPhantomLib/zubal_table.py
  SimindPhantomLib/logic.py
  SimindPhantomLib/lesion.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/Data/ts_to_zubal.csv
  Resources/Data/zubal_indices.csv
  Resources/Data/generic_120kVp_curve.csv
  )

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
  )

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
endif()
