// This source code is property of the Computer Graphics and Visualization chair
// of the TU Dresden. Do not distribute! Copyright (C) CGV TU Dresden - All
// Rights Reserved
//
// The main file of the plugin. It defines a class that demonstrates how to
// register with the scene graph, drawing primitives, creating a GUI, using a
// config file and various other parts of the framework.

// Framework core
#include "cgv/math/fvec.h"
#include <cgv/base/register.h>
#include <cgv/gui/provider.h>
#include <cgv/gui/trigger.h>
#include <cgv/math/ftransform.h>
#include <cgv/render/attribute_array_binding.h>
#include <cgv/render/drawable.h>
#include <cgv/render/frame_buffer.h>
#include <cgv/render/shader_program.h>
#include <cgv/render/texture.h>
#include <cgv/render/vertex_buffer.h>

// Framework standard libraries
#include <cgv_gl/gl/gl.h>

// Framework standard plugins
#include <plugins/cmf_tt_gl_font/tt_gl_font_server.h>
#include <random>

// Some constant symbols
#define FB_MAX_RESOLUTION 2048

// The CGV framework demonstration class
class bssrdf
    : public cgv::base::base,      // This class supports reflection
      public cgv::gui::provider,   // Instances of this class provde a GUI
      public cgv::render::drawable // Instances of this class can be rendered
{
protected:
  ////
  // Stuff we expose via reflection

  // Whether to use wireframe mode (helps visually debugging the custom
  // tesselation task)
  bool wireframe;

  // Internal stuff we don't expose via reflection

  cgv::render::shader_program bssrdf_shader;

  // Geometry buffers
  struct vertex {
    cgv::vec3 pos;
    cgv::vec2 tcoord;
  };

  struct strand_vertex {
    cgv::vec3 pos;
    cgv::vec3 tan;
    cgv::vec2 rootUV;
    float VAlong;
  };

  std::vector<vertex> vertices;
  std::vector<strand_vertex> strand_vertices;
  std::vector<uint32_t> strand_indices;

  cgv::render::vertex_buffer vb;
  cgv::render::attribute_array_binding vertex_array;

  // Flag for checking whether we have to reinit due to change in desired
  // offscreen framebuffer resolution
  bool fb_invalid;

public:
  // Default constructor
  bssrdf() : wireframe(false) {
    // Make sure the font server knows about the fonts packaged with the
    // exercise
    cgv::scan_fonts("./data/Fonts");
  }

  // Should be overwritten to sensibly implement the cgv::base::named interface
  std::string get_type_name(void) const { return "bssrdf"; }

  // Part of the cgv::base::base interface, can be implemented to make data
  // members of this class available as named properties, e.g. for use with
  // config files
  bool self_reflect(cgv::reflect::reflection_handler &rh) {

    // Task 1.1: make sure your quad tesselation toggle can be set via config
    //           file.
    // Reflect the properties
    return rh.reflect_member("wireframe", wireframe);
  }

  // Part of the cgv::base::base interface, should be implemented to respond to
  // write access to reflected data members of this class, e.g. from config file
  // processing or gui interaction.
  void on_set(void *member_ptr) {

    update_member(member_ptr);

    if (this->is_visible())
      post_redraw();
  }

  // We use this for validating GUI input
  bool gui_check_value(cgv::gui::control<int> &ctrl) { return true; }

  // We use this for acting upon validated GUI input
  void gui_value_changed(cgv::gui::control<int> &ctrl) {
    // Redraw the scene
    post_redraw();
  }

  // Required interface for cgv::gui::provider
  void create_gui(void) {}

  // Part of the cgv::render::drawable interface, can be overwritten if there is
  // some intialization work to be done that needs a set-up and ready graphics
  // context, which usually you don't have at object construction time. Should
  // return true if the initialization was successful, false otherwise.
  bool init(cgv::render::context &ctx) {
    // Keep track of success - do it this way (instead of e.g. returning false
    // immediatly) to perform every init step even if some go wrong.
    bool success = true;

    // Init geometry buffers
    // - get a reference to the default shader, from which we're going to query
    // named
    //   locations of the vertex layout
    // cgv::render::shader_program &default_shader =
    //     ctx.ref_default_shader_program(true /* true for texture support */);

    if (!bssrdf_shader.build_program(ctx, "bssrdf.glpr")) {
      std::cerr << "could not build the bssrdf shader program" << std::endl;
      exit(0);
    }
    // - generate actual geometry
    init_unit_square_geometry();
    // - obtain type descriptors for the automatic array binding facilities of
    // the
    //   framework
    cgv::render::type_descriptor
        vec2type = cgv::render::element_descriptor_traits<
            cgv::vec2>::get_type_descriptor(vertices[0].tcoord),
        vec3type = cgv::render::element_descriptor_traits<
            cgv::vec3>::get_type_descriptor(vertices[0].pos);
    // - create buffer objects
    success = vb.create(ctx, &(vertices[0]), vertices.size()) && success;
    success = vertex_array.create(ctx) && success;
    success = vertex_array.set_attribute_array(
                  ctx, bssrdf_shader.get_position_index(), vec3type, vb,
                  0, // position is at start of the struct <-> offset = 0
                  vertices.size(), // number of position elements in the array
                  sizeof(vertex)   // stride from one element to next
                  ) &&
              success;
    success = vertex_array.set_attribute_array(
                  ctx, 1, vec2type, vb,
                  sizeof(cgv::vec3), // tex coords follow after position
                  vertices.size(),   // number of texcoord elements in the array
                  sizeof(vertex)     // stride from one element to next
                  ) &&
              success;

    // Flag offscreen framebuffer as taken care of
    fb_invalid = false;

    // All initialization has been attempted
    return success;
  }

  // Part of the cgv::render::drawable interface, can be overwritten if there is
  // some work to be done before actually rendering a frame.
  void init_frame(cgv::render::context &ctx) {}

  // Should be overwritten to sensibly implement the cgv::render::drawable
  // interface
  void draw(cgv::render::context &ctx) {
    // Observe wireframe mode
    glPushAttrib(GL_POLYGON_BIT);
    if (wireframe)
      glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

    // Shortcut to the built-in default shader with lighting and texture support

    // Enable shader program we want to use for drawing
    bssrdf_shader.enable(ctx);

    // Set the "color" vertex attribute for all geometry drawn hereafter, except
    // if it explicitely specifies its own color data by means of an attribute
    // array. Note that this only works for shaders that define a vec4 attribute
    // named "color" in their layout specification. We want white to retain the
    // original color information in the texture.
    ctx.set_color(cgv::rgb(1.0f));

    // Draw the node's scene geometry - save current modelview matrix because we
    // have some node-internal transformations that we do not want to spill over
    // to other drawables.
    ctx.push_modelview_matrix();
    // // Account for aspect ratio of the offscreen texture
    // ctx.mul_modelview_matrix(cgv::math::rotate4(90.0, 0.0, 0.0, 0.0));

    ctx.mul_modelview_matrix(cgv::math::translate4(0.0, -0.1, 0.0));
    ctx.mul_modelview_matrix(cgv::math::rotate4(-90.0, 1.0, 0.0, 0.0));

    // Draw front side
    //*********************************************************************/
    // Task 1.1: If enabled, render the quad with custom tesselation
    //           instead of using tesselate_unit_square(). You can invoke
    //           the method draw_my_unit_square() for this.
    ctx.tesselate_unit_square();

    //*****************************************************************/
    glPopAttrib();
    ctx.pop_modelview_matrix();

    // Disable shader program and texture
    bssrdf_shader.disable(ctx);
  }

  // Creates the custom geometry for the quad
  void init_unit_square_geometry(void) {
    // Prepare array
    vertices.resize(4);
    // lower-left
    vertices[0].pos.set(-1, -1, 0);
    vertices[0].tcoord.set(0, 0);
    // lower-right
    vertices[1].pos.set(1, -1, 0);
    vertices[1].tcoord.set(1, 0);
    // top-left
    vertices[2].pos.set(-1, 1, 0);
    vertices[2].tcoord.set(0, 1);
    // top-right
    vertices[3].pos.set(1, 1, 0);
    vertices[3].tcoord.set(1, 1);
  }

  void init_strands_geometry(void) {
    const uint32_t STRAND_AMOUNT = 10000;
    const float STRAND_TOTAL_LENGTH = 0.1;
    const uint32_t STRAND_SEGMENTS_LENGTH = 4;
    const uint32_t STRAND_SEGMENTS_CIRCUMFERENCE = 8;
    const float STRAND_RADIUS = 0.0015f;

    strand_vertices.reserve(STRAND_AMOUNT * (STRAND_SEGMENTS_LENGTH + 1) *
                            STRAND_SEGMENTS_CIRCUMFERENCE);
    strand_indices.reserve(STRAND_AMOUNT * STRAND_SEGMENTS_LENGTH *
                           STRAND_SEGMENTS_CIRCUMFERENCE * 6);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
    std::uniform_real_distribution<float> tiltDist(-0.2f, 0.2f);

    auto sampleRootUV = [&]() -> cgv::vec2 {
      return cgv::vec2(dist01(rng), dist01(rng)); // in [0,1]^2
    };

    auto quadPosFromUV = [&](const cgv::vec2 &uv) -> cgv::vec3 {
      float x = uv.x() - 0.5f;
      float z = uv.y() - 0.5f;
      return cgv::vec3(x, 0.0f, z);
    };

    for (uint32_t s = 0; s < STRAND_AMOUNT; s++) {
      // === Per-strand data ===

      cgv::vec2 rootUV = sampleRootUV();

      cgv::vec3 rootPos = quadPosFromUV(rootUV);

      cgv::vec3 T =
          cgv::math::normalize(cgv::vec3(tiltDist(rng), 1.0f, tiltDist(rng)));

      cgv::vec3 arbitrary =
          std::abs(T.y()) < 0.9f ? cgv::vec3(0, 1, 0) : cgv::vec3(1, 0, 0);

      cgv::vec3 B = cgv::math::normalize(cgv::math::cross(T, arbitrary));

      cgv::vec3 N = cgv::math::normalize(cgv::math::cross(B, T));

      uint32_t baseVertexIndex = static_cast<uint32_t>(strand_vertices.size());

      for (uint32_t i = 0; i <= STRAND_SEGMENTS_LENGTH; i++) {
        float t = (float)i / (float)STRAND_SEGMENTS_LENGTH;
        float segmentHeight = STRAND_SEGMENTS_LENGTH * t;

        cgv::vec3 center = rootPos + T * segmentHeight;

        for (uint32_t j = 0; j < STRAND_SEGMENTS_CIRCUMFERENCE; ++j) {
        }
      }
    }

    strand_vertex v{};
  }

  // Draw method for a custom quad
  void draw_my_unit_square(cgv::render::context &ctx) {
    vertex_array.enable(ctx);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, (GLsizei)vertices.size());
    vertex_array.disable(ctx);
  }
};

// Create an instance of the demo class at plugin load and register it with the
// framework
cgv::base::object_registration<bssrdf> bssrdf_registration("");

// The following could be used to register the class with the framework but NOT
// create it upon plugin load. Instead, the user can create an instance from the
// application menu. However, config files are not straight-forward to use in
// this case, which is why we go for the method above.
/*
        cgv::base::factory_registration<cgv_demo> cgv_demo_factory(
                "new/cgv_demo", // menu path
                'D',            // the shortcut - capital D means ctrl+d
                true            // whether the class is supposed to be a
   singleton
        );
*/
