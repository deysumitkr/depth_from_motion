#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include <cstdio>

#include "pointCloud.h"

float DEL_X = 0;
float DEL_Y = 0;

float ZOOM_Z = 0;

float STRAFE_X = 0;
float STRAFE_Y = 0;



void controls(GLFWwindow* window, int key, int scancode, int action, int mods){
    if(action == GLFW_PRESS){
        
        if(key == GLFW_KEY_ESCAPE){
            glfwSetWindowShouldClose(window, GL_TRUE);
        }
        
        if(key == GLFW_KEY_RIGHT) ++DEL_X;
        if(key == GLFW_KEY_LEFT) --DEL_X;
        if(key == GLFW_KEY_UP) ++DEL_Y;
        if(key == GLFW_KEY_DOWN) --DEL_Y;
        
        if(key == GLFW_KEY_KP_ADD) ZOOM_Z+=0.05;
        if(key == GLFW_KEY_KP_SUBTRACT) ZOOM_Z-=0.05;
        
        if(key == GLFW_KEY_A) STRAFE_X-=0.05;
        if(key == GLFW_KEY_D) STRAFE_X+=0.05;
        if(key == GLFW_KEY_W) STRAFE_Y-=0.05;
        if(key == GLFW_KEY_S) STRAFE_Y+=0.05;
	}
	
	if(action == GLFW_RELEASE){
		if(key == GLFW_KEY_RIGHT) DEL_X=0;
        if(key == GLFW_KEY_LEFT) DEL_X=0;
        if(key == GLFW_KEY_UP) DEL_Y=0;
        if(key == GLFW_KEY_DOWN) DEL_Y=0;
        
        if(key == GLFW_KEY_KP_ADD) ZOOM_Z=0;
        if(key == GLFW_KEY_KP_SUBTRACT) ZOOM_Z=0;
        
        if(key == GLFW_KEY_A) STRAFE_X=0.0;
        if(key == GLFW_KEY_D) STRAFE_X=0.0;
        if(key == GLFW_KEY_W) STRAFE_Y=0.0;
        if(key == GLFW_KEY_S) STRAFE_Y=0.0;
	}
}


GLFWwindow* initWindow(const int resX, const int resY){
    if(!glfwInit()){
        fprintf(stderr, "Failed to initialize GLFW\n");
        return NULL;
    }
    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing

    // Open a window and create its OpenGL context
    GLFWwindow* window = glfwCreateWindow(resX, resY, "Point Cloud | OpenGL", NULL, NULL);

    if(window == NULL){
        fprintf(stderr, "Failed to open GLFW window.\n");
        glfwTerminate();
        return NULL;
    }

    glfwMakeContextCurrent(window);
    glfwSetKeyCallback(window, controls);

    // Get info of GPU and supported OpenGL version
    printf("Renderer: %s\n", glGetString(GL_RENDERER));
    printf("OpenGL version supported %s\n", glGetString(GL_VERSION));

    glEnable(GL_DEPTH_TEST); // Depth Testing
    glDepthFunc(GL_LEQUAL);
    glDisable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    return window;
}

void drawCube(float p3D[], float c3D[], int len){
    
 
    static float preX=0, preY=0;
	preX += DEL_X;
	preY += DEL_Y;
	glRotatef(preX, 0, 1.0, 0);
	glRotatef(preY, 1.0, 0, 0);

    /* We have a color array and a vertex array */
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, p3D);
    glColorPointer(3, GL_FLOAT, 0, c3D);
    

    /* Send data : len number of vertices */
    glDrawArrays(GL_POINTS, 0, len);
    glPointSize(3.0);

    /* Cleanup states */
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    //alpha += 1;
}


void display( GLFWwindow* window, float p3D[], float c3D[], int len){
    while(!glfwWindowShouldClose(window)){
        // Scale to window size
        GLint windowWidth, windowHeight;
        glfwGetWindowSize(window, &windowWidth, &windowHeight);
        glViewport(0, 0, windowWidth, windowHeight);

        // Draw stuff
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION_MATRIX);
        glLoadIdentity();
        gluPerspective( 60, (double)windowWidth / (double)windowHeight, 0.1, 10000 );

        glMatrixMode(GL_MODELVIEW_MATRIX);
        
        static float preZ=-5;
		static float moveX = 0;
		static float moveY = 0;
        preZ += ZOOM_Z;
        moveX += STRAFE_X;
        moveY += STRAFE_Y;
        glTranslatef(moveX,moveY,preZ);
                
        drawCube(p3D, c3D, len);
        
        // Update Screen
        glfwSwapBuffers(window);

        // Check for any input, or window movement
        glfwPollEvents();
    }
}

int plotCloud(float p3D[], float c3D[], int len){
    GLFWwindow* window = initWindow(1024, 620);
    if( NULL != window )
    {
        display( window, p3D, c3D, len );
    }
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
