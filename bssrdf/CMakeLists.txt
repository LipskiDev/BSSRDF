
# compile a list of source files for each specific source type the CGV CMake build system knows about
set(SOURCES
	bssrdf.cxx
)
set(HEADERS
)

set(SHADERS
    glsl/bssrdf.glpr
    glsl/bssrdf.glvs
    glsl/bssrdf.glfs
)

# add our target to the CGV CMake build system
cgv_add_target(bssrdf
	TYPE plugin    NO_EXECUTABLE
	SOURCES        ${SOURCES}
	HEADERS        ${HEADERS}
    SHADER_SOURCES ${SHADERS}

	DEPENDENCIES
		cgv_viewer glew fltk cgv_utils cgv_type cgv_reflect cgv_data cgv_signal cgv_base cgv_media cgv_gui
		cgv_render cgv_os cgv_gl cg_fltk cmf_tt_gl_font crg_stereo_view cg_ext cg_meta cmi_io

	LINKTIME_PLUGIN_DEPENDENCIES
		cmf_tt_gl_font

	ADDITIONAL_CMDLINE_ARGS
		"config:\"${CMAKE_CURRENT_LIST_DIR}/config.def\""
)
