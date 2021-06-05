#!/usr/bin/env python3
# Renders a 2D model into a PPM image
import sys
import numpy as np

# ---------- Configuration types and constants ----------

MAX_SIZE = 1024
MAX_VAL = 255
MAX_LINE_LEN = 10240-1 # 10240 characters minus the \0 terminator
DEFAULT_BACKGROUND = 255
CHANNELS_N = 3
COORD_N = 3
DEFAULT_COLOR = (0, 0, 0,)
IMAGE_DTYPE = np.uint8
VIEWPORT_DTYPE = np.int64
MODEL_DTYPE = np.float64
ZBUFFER_DTYPE = np.float64
ZBUFFER_BACKGROUND = -np.inf
COORD_DTYPE = np.int64

# ---------- Output routines ----------

def put_string(output, output_file):
    output = output.encode('ascii') if isinstance(output, str) else output
    written_n = output_file.write(output)
    if written_n != len(output):
        print('error writing to output stream', file=sys.stderr)
        sys.exit(1)

def save_ppm(image, output_file):
    # Defines image header
    magic_number_1 = 'P'
    magic_number_2 = '6'
    width  = image.shape[1]
    height = image.shape[0]
    end_of_header = '\n'

    # Writes header
    put_string(magic_number_1, output_file)
    put_string(magic_number_2, output_file)
    put_string('\n', output_file)
    put_string('%d %d\n' % (width, height), output_file)
    put_string('%d' % MAX_VAL, output_file)
    put_string(end_of_header, output_file)

    # Outputs image
    put_string(image.tobytes(), output_file)

# ---------- Drawing/model routines ----------

def adjust_points(coordinates, points_n):
    # ...converts to points
    points = np.reshape(coordinates, (points_n, 3,))
    # ...converts points to homogeneous coordinates
    points = np.c_[points, np.ones(points_n)]
    # ...applies transformation matrix
    points = np.matmul(transform, points.T).T
    # ...saves pre projection state points for z buffer
    z_points = points
    # ...applies viewport matrix
    points = np.matmul(projection, points.T).T
    # ...renormalizes points
    points = (points.T / points[:,3]).T
    # ...rounds points to the nearest integer and converts to integer data type
    points = np.rint(points).astype(COORD_DTYPE)

    return points, z_points


def draw_line(image, x0, y0, z0, x1, y1, z1, color):
    h, w, _ = image.shape
    # Translate to fit viewport
    x0+=int(w/2)
    y0+=int(h/2)
    x1+=int(w/2)
    y1+=int(h/2)
    # Computes differences
    dx = x1-x0
    dy = y1-y0
    dc = abs(dx) # delta x in book - here we are using row, col coordinates
    dr = abs(dy) # delta y in book
    if dr <= dc:
        # Line inclination is at most 1
        # Swaps points if c1<c0 and converts x,y coordinates to row,col coordinates
        # dx>=0 => x1>=x0 => c1>=x0
        r0 = h-1-y0 if dx>=0 else h-1-y1
        r1 = h-1-y1 if dx>=0 else h-1-y0
        c0 =     x0 if dx>=0 else x1
        c1 =     x1 if dx>=0 else x0
        # Implements Bresenham's midpoint algorithm for lines
        # ...deltas of Bressenham's algorithm
        d_horizontal = 2*dr      # delta east in book
        d_diagonal   = 2*(dr-dc) # delta northeast in book
        # ...draws line
        pixel_r = r0
        step_row = 1 if r1>=r0 else -1
        d = 2*dr - dc # starting D value, D_init in book
        for pixel_c in range(c0, c1+1):
            if(0 <= pixel_r < h and 0 <= pixel_c < w): # Clipping strategy
                if(c1!=c0):
                    q = np.float64(abs((pixel_c-c0)/(c1-c0)))
                    z = (z0)*(1-q)+(z1)*q # Interpolation for z buffer
                else:
                    z = max(z0,z1) # c1=c0 implies that a line perpendicular to plane is drawn
                if(z > zbuffer[pixel_r, pixel_c]): # Z buffer algorithm
                    zbuffer[pixel_r , pixel_c ] = z
                    image[pixel_r, pixel_c, :] = color
            if d<=0:
                d += d_horizontal
            else:
                d += d_diagonal
                pixel_r += step_row
    else:
        # Line inclination is greater than one -- inverts the roles of row and column
        # Swaps points if y1>y0 and converts x,y coordinates to row,col coordinates
        # dy<=0 => y1<=y0 => r1>=r0
        r0 = h-1-y0 if dy<=0 else h-1-y1
        r1 = h-1-y1 if dy<=0 else h-1-y0
        c0 =     x0 if dy<=0 else x1
        c1 =     x1 if dy<=0 else x0
        # Implements Bresenham's midpoint algorithm for lines
        # ...deltas of Bressenham's algorithm - same as above, but with coordinates inverted
        d_vertical = 2*dc
        d_diagonal = 2*(dc-dr)
        pixel_r = r0
        pixel_c = c0
        step_col = 1 if c1>=c0 else -1
        d = 2*dc - dr # starting D value, D_init in book
        for pixel_r in range(r0, r1+1):
            if(0 <= pixel_r < h and 0 <= pixel_c < w): # Clipping strategy
                if(r1!=r0):
                    q = np.float64(abs((pixel_r-r0)/(r1-r0)))
                    z = (z0)*(1-q)+(z1)*q # Interpolation for z buffer
                else:
                    z = max(z0,z1) # r1=r0 implies that a line perpendicular to plane is drawn
                if(z > zbuffer[pixel_r , pixel_c ]): # Z buffer algorithm
                    zbuffer[pixel_r, pixel_c] = z
                    image[pixel_r, pixel_c, :] = color
            if (d<=0):
                d += d_vertical
            else:
                d += d_diagonal
                pixel_c += step_col

# ---------- Main routine ----------

# Parses and checks command-line arguments
if len(sys.argv)!=3:
    print("usage: python draw_2d_model.py <input.dat> <output.ppm>\n"
          "       interprets the drawing instructions in the input file and renders\n"
          "       the output in the NETPBM PPM format into output.ppm")
    sys.exit(1)

input_file_name  = sys.argv[1]
output_file_name = sys.argv[2]

# Reads input file and parses its header
with open(input_file_name, 'rt', encoding='utf-8') as input_file:
    input_lines = input_file.readlines()

if input_lines[0] != 'EA979V4\n':
    print(f'input file format not recognized!', file=sys.stderr)
    sys.exit(1)

dimensions = input_lines[1].split()
width = int(dimensions[0])
height = int(dimensions[1])

if width<=0 or width>MAX_SIZE or height<=0 or height>MAX_SIZE:
    print(f'input file has invalid image dimensions: must be >0 and <={MAX_SIZE}!', file=sys.stderr)
    sys.exit(1)

# Creates image
image = np.full((height, width, CHANNELS_N), fill_value=DEFAULT_BACKGROUND, dtype=IMAGE_DTYPE)
zbuffer = np.full((height, width,), fill_value=ZBUFFER_BACKGROUND, dtype=ZBUFFER_DTYPE)

# General porposes variables
color = np.array(DEFAULT_COLOR, dtype=IMAGE_DTYPE)
transform = np.eye(CHANNELS_N+1, dtype=MODEL_DTYPE)
projection = np.eye(CHANNELS_N+1, dtype=MODEL_DTYPE) # The default projection is the orthogonal one
stack = []


# Main loop - interprets and renders drawing commands
for line_n,line in enumerate(input_lines[2:], start=3):

    if len(line)>MAX_LINE_LEN:
        print(f'line {line_n}: line too long!', file=sys.stderr)
        sys.exit(1)

    if not line.strip():
        # Blank line - skips
        continue
    if line[0] == '#':
        # Comment line - skips
        continue

    tokens = line.strip().split()
    command = tokens[0]
    parameters = tokens[1:]
    def check_parameters(n):
        if len(parameters) != n:
            print(f'line {line_n}: command {command} expected {n} parameters but got {len(parameters)}!',
                  file=sys.stderr)
            sys.exit(1)

    if command == 'c':
        # Clears with new background color
        check_parameters(CHANNELS_N)
        background_color = np.array(parameters, dtype=IMAGE_DTYPE)
        image[...] = background_color
        zbuffer[...] = ZBUFFER_BACKGROUND

    elif command == 'V':
        # Replace projection matrix
        check_parameters(16)
        projection = np.array(parameters, dtype=MODEL_DTYPE).reshape((4,4,))

    elif command == 'M':
        # Replace transformation matrix
        check_parameters(16)
        transform = np.array(parameters, dtype=MODEL_DTYPE).reshape((4,4,))

    elif command in 'LPR':
        # L - line P- poliline R - Poligon
        if command == 'L':
            check_parameters(6)
            points_n = 2
            coordinates = np.array(parameters, dtype=MODEL_DTYPE)
        else:
            points_n = int(parameters[0])
            check_parameters(points_n*3 + 1)
            coordinates = np.array(parameters[1:], dtype=MODEL_DTYPE)

        # Receive points for draw line
        points, z_points = adjust_points(coordinates, points_n)

        # ...makes drawing
        for i in range(1, points_n):
            draw_line(image, points[i-1,0], points[i-1,1], z_points[i-1,2],
                      points[i,0], points[i,1], z_points[i,2], color)
        if command == 'R':
            draw_line(image, points[points_n-1,0], points[points_n-1,1], z_points[points_n-1,2],
                      points[0,0], points[0,1],z_points[0,2], color)

    elif command == 'C':
        # Pen color
        check_parameters(CHANNELS_N)
        color = np.array(parameters, dtype=IMAGE_DTYPE)

    elif command == 'PUSH':
        # Put current transform matrix on the stack
        stack.append(transform)

    elif command == 'POP':
        # Remove top of stack
        transform = np.array(stack.pop())

    elif command == 'm':
        # Left multiplication of transformation matrix
        check_parameters(16)
        transform_acc = np.array(parameters, dtype=MODEL_DTYPE).reshape((4,4,))
        transform = np.matmul(transform, transform_acc)

    elif command == 'SPH':
        # Esfera om meridianos e e paralelos
        radium = float(parameters[0])
        meridians = int(parameters[1])
        parallels = int(parameters[2])
        pi = np.pi
        cos = np.cos
        sin = np.sin
        coordinates_parallels = []
        coordinates_meridians = []
        points_parallels =[]
        points_meridians = []
        # drawing parallels
        for i in range(parallels+1):
            phi = (i*np.pi)/(parallels+1)
            for j in range (meridians):
                theta = (j*np.pi*2)/(meridians)

                x = radium*sin(phi)*cos(theta)
                z = radium*sin(phi)*sin(theta)
                y = radium*cos(phi)

                listofPoints = [x.real,y.real,z.real]

                coordinates_parallels.append(listofPoints)
        points_parallels, z_points = adjust_points(coordinates_parallels, len(coordinates_parallels))
        for i in range(parallels+1): # run through the number of parallels
            for j in range(meridians-1): # the number of parallels define the number of points for each meridian
                draw_line(image, points_parallels[j + meridians*i, 0], points_parallels[j + meridians*i, 1], z_points[j + meridians*i][2], points_parallels[j + meridians*i + 1, 0], points_parallels[j + meridians*i + 1, 1], z_points[j + meridians*i + 1][2], color)
                if j == meridians -2: #closes the parallels, connecting the last and initial points
                    draw_line(image, points_parallels[j + meridians*i + 1, 0], points_parallels[j+meridians*i + 1, 1],
                              z_points[j + meridians*i + 1][2], points_parallels[j+meridians*i-meridians + 2, 0],
                              points_parallels[j+meridians*i-meridians + 2 , 1], z_points[j+meridians*i-meridians + 2][2], color)

        #drawing meridians
        for i in range(2*meridians+1):
            theta = (i*np.pi*2)/(2*meridians)

            for j in range (parallels+2): #(0,1,2,3,4,5), (6,7,8,9,10,11)
                phi = (j*np.pi)/(parallels+1)

                z = radium*sin(phi)*cos(theta)
                x = radium*sin(theta)*sin(phi)
                y = radium*cos(phi)

                listofPoints = [x.real,y.real,z.real]

                coordinates_meridians.append(listofPoints)
        points_meridians, z_points = adjust_points(coordinates_meridians, len(coordinates_meridians))
        for i in range(2*meridians+1): # run through the number of meridians
            for j in range(parallels+1): # the number of parallels define the number of points for each meridian
                draw_line(image, points_meridians[j + (parallels+2)*i, 0], points_meridians[j + (parallels+2)*i, 1],
                          z_points[j + (parallels+2)*i][2], points_meridians[j + (parallels+2)*i + 1, 0],
                          points_meridians[j + (parallels+2)*i + 1, 1], z_points[j + (parallels+2)*i + 1][2], color)
        continue

    elif command == 'CUB':
        # Cube drawing

        check_parameters(2)
        diameter = np.array(parameters[0], dtype=MODEL_DTYPE)
        radius = int(diameter/2)
        with_diagonals = int(parameters[1])

        square = [[ radius  ,  radius , -radius], [-radius ,  radius , -radius],
                  [-radius  , -radius , -radius], [ radius , -radius , -radius],
                  [ radius  ,  radius ,  radius], [-radius ,  radius ,  radius],
                  [-radius  , -radius ,  radius], [ radius , -radius ,  radius]]

        points, z_points = adjust_points(square, len(square))

        for i in range(8):
            for j in range(i+1,8):
                if ((square[i][0] == square[j][0] and square[i][1] == square[j][1]
                     and square[i][2] != square[j][2])
                or  (square[i][0] != square[j][0] and square[i][1] == square[j][1]
                     and square[i][2] == square[j][2])
                or  (square[i][0] == square[j][0] and square[i][1] != square[j][1]
                     and square[i][2] == square[j][2])):
                    draw_line(image, points[i, 0], points[i, 1], z_points[i,2],
                              points[j, 0], points[j, 1], z_points[j,2], color)

        if (with_diagonals):
            for i in range(8):
                for j in range(i+1,8):
                    if ((square[i][0] != square[j][0] and square[i][1] == square[j][1]
                         and square[i][2] != square[j][2])
                    or  (square[i][0] != square[j][0] and square[i][1] != square[j][1]
                         and square[i][2] == square[j][2])
                    or  (square[i][0] == square[j][0] and square[i][1] != square[j][1]
                         and square[i][2] != square[j][2])):
                        draw_line(image, points[i, 0], points[i, 1], z_points[i,2],
                                  points[j, 0], points[j, 1], z_points[j,2], color)

    else:
        print(f'line {line_n}: unrecognized command "{command}"!', file=sys.stderr)
        sys.exit(1)

# If we reached this point, everything went well - outputs rendered image file
with open(output_file_name, 'wb') as output_file:
    save_ppm(image, output_file)
