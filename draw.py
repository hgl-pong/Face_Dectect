import glfw
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import numpy as np
import pyrr
from PIL import Image

vertex_src = """
# version 330

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;
layout(location = 2) in vec2 a_texture;

uniform mat4 rotation;

out vec3 v_color;
out vec2 v_texture;

void main()
{
    gl_Position = rotation * vec4(a_position, 1.0);
    v_color = a_color;
    v_texture = a_texture;
    
    //v_texture = 1 - a_texture;                      // Flips the texture vertically and horizontally
    //v_texture = vec2(a_texture.s, 1 - a_texture.t); // Flips the texture vertically
}
"""

fragment_src = """
# version 330

in vec2 v_texture;
out vec4 out_color;
uniform sampler2D s_texture;

void main()
{
    out_color = texture(s_texture, v_texture) ;
}
"""

# glfw callback functions
def window_resize(window, width, height):
    glViewport(0, 0, width, height)

# initializing glfw library
if not glfw.init():
    raise Exception("glfw can not be initialized!")

# creating the window
window = glfw.create_window(800, 500, "Face_Detection", None, None)

# check if window was created
if not window:
    glfw.terminate()
    raise Exception("glfw window can not be created!")

# set window's position
glfw.set_window_pos(window, 100, 100)

# set the callback function for window resize
glfw.set_window_size_callback(window, window_resize)

# make the context current
glfw.make_context_current(window)

vertices = [-1, -1,  1,  1.0, 0.0, 0.0,  0.0, 0.0,
             1, -1,  1,  0.0, 1.0, 0.0,  1.0, 0.0,
             1,  1,  1,  0.0, 0.0, 1.0,  1.0, 1.0,
            -1, 1,  1,  1.0, 1.0, 1.0,  0.0, 1.0]

indices = [0,  1,  2,  2,  3,  0]

vertices = np.array(vertices, dtype=np.float32)
indices = np.array(indices, dtype=np.uint32)

shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER), compileShader(fragment_src, GL_FRAGMENT_SHADER))

# Step2: 创建并绑定VBO 对象 传送数据
VBO = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, VBO)
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)

# Step3: 创建并绑定EBO 对象 传送数据
EBO = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
# Step4: 指定解析方式  并启用顶点属性
# 顶点位置属性 
glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(0))
# 顶点颜色属性
glEnableVertexAttribArray(1)
glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(12))
# 顶点纹理属性
glEnableVertexAttribArray(2)
glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, vertices.itemsize * 8, ctypes.c_void_p(24))

texture = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D, texture)

# Set the texture wrapping parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
# Set texture filtering parameters
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

# load image
image = Image.open("0001.jpg")
image = image.transpose(Image.FLIP_TOP_BOTTOM)
img_data = image.convert("RGBA").tobytes()
# img_data = np.array(image.getdata(), np.uint8) # second way of getting the raw image data
glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

glUseProgram(shader)
glClearColor(1, 1, 1, 1)
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
model = pyrr.matrix44.create_from_translation(pyrr.Vector3([0.0, 0.0, 0]))
rotation_loc = glGetUniformLocation(shader, "rotation")

# the main application loop
while not glfw.window_should_close(window):
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

    # 绘制第一个矩形
    glUniformMatrix4fv(rotation_loc, 1, GL_FALSE, model)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, None)

    glfw.swap_buffers(window)

# terminate glfw, free up allocated resources
glfw.terminate()

