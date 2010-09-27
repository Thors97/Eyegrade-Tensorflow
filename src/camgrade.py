import pygame
import Image
from pygame.locals import *
import sys
import ConfigParser, os
from optparse import OptionParser
import opencv
#this is important for capturing/displaying images
from opencv import highgui 
import imageproc
import time
import copy

program_name = 'camgrade'
version = '0.1.1'
version_status = 'alpha'

class Exam(object):
    def __init__(self, image, model, solutions, valid_student_ids = None,
                 im_id = None, save_stats = False):
        self.image = image
        self.model = model
        self.solutions = solutions
        self.im_id = im_id
        self.correct = None
        self.score = None
        self.original_decisions = copy.copy(self.image.decisions)
        self.save_stats = save_stats
        if self.image.options['read-id']:
            self.student_id = self.decide_student_id(valid_student_ids)
            self.student_id_filter = []
        else:
            self.student_id = '-1'

    def grade(self):
        good = 0
        bad = 0
        undet = 0
        self.correct = []
        for i in range(0, len(self.image.decisions)):
            if self.image.decisions[i] > 0:
                if self.solutions[i] == self.image.decisions[i]:
                    good += 1
                    self.correct.append(True)
                else:
                    bad += 1
                    self.correct.append(False)
            elif self.image.decisions[i] < 0:
                undet += 1
                self.correct.append(False)
            else:
                self.correct.append(False)
        self.score = (good, bad, undet)

    def draw_answers(self):
        good, bad, undet = self.score
        self.image.draw_answers(self.solutions, self.model, self.correct,
                                self.score[0], self.score[1], self.score[2],
                                self.im_id)

    def save_image(self, filename_pattern):
        highgui.cvSaveImage(filename_pattern%self.im_id, self.image.image_drawn)

    def save_debug_images(self, filename_pattern):
        raw_pattern = filename_pattern + "-raw"
        proc_pattern = filename_pattern + "-proc"
        highgui.cvSaveImage(raw_pattern%self.im_id, self.image.image_raw)
        highgui.cvSaveImage(proc_pattern%self.im_id, self.image.image_proc)

    def save_answers(self, answers_file, stats = None):
        sep = "\t"
        f = open(answers_file, "a")
        f.write(str(self.im_id))
        f.write(sep)
        f.write(self.student_id)
        f.write(sep)
        f.write(str(self.model))
        f.write(sep)
        f.write(str(self.score[0]))
        f.write(sep)
        f.write(str(self.score[1]))
        f.write(sep)
        f.write(str(self.score[2]))
        f.write(sep)
        f.write("/".join([str(d) for d in self.image.decisions]))
        if stats is not None and self.save_stats:
            f.write(sep)
            f.write(str(stats['time']))
            f.write(sep)
            f.write(str(stats['manual-changes']))
            f.write(sep)
            f.write(str(stats['num-captures']))
            f.write(sep)
            f.write(str(stats['num-student-id-changes']))
            f.write(sep)
            f.write(str(stats['id-ocr-digits-total']))
            f.write(sep)
            f.write(str(stats['id-ocr-digits-error']))
            f.write(sep)
            f.write(stats['id-ocr-detected'])
        f.write('\n')
        f.close()

    def toggle_answer(self, question, answer):
        if self.image.decisions[question] == answer:
            self.image.decisions[question] = 0
        else:
            self.image.decisions[question] = answer
        self.grade()
        self.image.clean_drawn_image()
        self.draw_answers()

    def invalidate_id(self):
        self.__update_student_id(None)

    def num_manual_changes(self):
        return len([d1 for (d1, d2) in \
                        zip(self.original_decisions, self.image.decisions) \
                        if d1 != d2])

    def decide_student_id(self, valid_student_ids):
        student_id = '-1'
        self.ids_rank = None
        if self.image.id is not None:
            if valid_student_ids is not None:
                ids_rank = [(self.__id_rank(sid, self.image.id_scores), sid) \
                                for sid in valid_student_ids]
                self.ids_rank = sorted(ids_rank, reverse = True)
                student_id = self.ids_rank[0][1]
                self.image.id = student_id
                self.ids_rank_pos = 0
            else:
                student_id = self.image.id
        return student_id

    def try_next_student_id(self):
        if self.ids_rank is not None \
                and self.ids_rank_pos < len(self.ids_rank) - 1:
            self.ids_rank_pos += 1
            self.__update_student_id(self.ids_rank[self.ids_rank_pos][1])

    def filter_student_id(self, digit):
        self.student_id_filter.append(digit)
        ids = [sid for sid in self.ids_rank \
                   if ''.join(self.student_id_filter) in sid[1]]
        if len(ids) > 0:
            self.__update_student_id(ids[0][1])
        else:
            self.__update_student_id(None)
            self.student_id_filter = []
            self.ids_rank_pos = -1

    def reset_student_id_editor(self):
        self.student_id_filter = []
        self.ids_rank_pos = 0
        self.__update_student_id(self.ids_rank[0][1])

    def __id_rank(self, student_id, scores):
        rank = 0.0
        for i in range(len(student_id)):
            rank += scores[i][int(student_id[i])]
        return rank

    def __update_student_id(self, new_id):
        if new_id is None or new_id == '-1':
            self.image.id = None
            self.student_id = '-1'
        else:
            self.image.id = new_id
            self.student_id = new_id
        self.image.clean_drawn_image()
        self.draw_answers()

class PerformanceProfiler(object):
    def __init__(self):
        self.start()
        self.num_captures = 0
        self.num_student_id_changes = 0

    def start(self):
        self.time0 = time.time()

    def count_capture(self):
        self.num_captures += 1

    def count_student_id_change(self):
        self.num_student_id_changes += 1

    def finish_exam(self, exam):
        time1 = time.time()
        stats = {}
        stats['time'] = time1 - self.time0
        stats['manual-changes'] = exam.num_manual_changes()
        stats['num-captures'] = self.num_captures
        stats['num-student-id-changes'] = self.num_student_id_changes
        self.compute_ocr_stats(stats, exam)
        self.time0 = time1
        self.num_captures = 0
        self.num_student_id_changes = 0
        return stats

    def compute_ocr_stats(self, stats, exam):
        if exam.image.id is None:
            digits_total = 0
            digits_error = 0
        else:
            digits_total = len(exam.image.id)
            digits_error = len([1 for a, b in zip(exam.image.id,
                                                  exam.image.id_ocr_original) \
                                    if a != b])
        stats['id-ocr-digits-total'] = digits_total
        stats['id-ocr-digits-error'] = digits_error
        if exam.image.id_ocr_original is not None:
            stats['id-ocr-detected'] = exam.image.id_ocr_original
        else:
            stats['id-ocr-detected'] = '-1'

def init(camera_dev):
    camera = imageproc.init_camera(camera_dev)
    return camera

def process_exam_data(filename):
    exam_data = ConfigParser.SafeConfigParser()
    exam_data.read([filename])
    try:
        num_models = exam_data.getint("exam", "num-models")
    except:
        num_models = 1
    try:
        id_num_digits = exam_data.getint("exam", "id-num-digits")
    except:
        id_num_digits = 0
    solutions = []
    for i in range(0, num_models):
        key = "model-" + chr(65 + i)
        solutions.append(parse_solutions(exam_data.get("solutions", key)))
    dimensions = parse_dimensions(exam_data.get("exam", "dimensions"))
    return solutions, dimensions, id_num_digits

def parse_solutions(s):
    return [int(num) for num in s.split("/")]

def parse_dimensions(s):
    dimensions = []
    boxes = s.split(";")
    for box in boxes:
        dims = box.split(",")
        dimensions.append((int(dims[0]), int(dims[1])))
    return dimensions

def decode_model_2x31(bits):
    # x3 = x0 ^ x1 ^ not x2; x0-x3 == x4-x7
    valid = False
    if len(bits) == 3:
        valid = True
    elif len(bits) >= 4:
        if (bits[3] == bits[0] ^ bits[1] ^ (not bits[2])):
            if len(bits) < 8:
                valid = True
            else:
                valid = (bits[0:4] == bits[4:8])
    if valid:
        return bits[0] | bits[1] << 1 | bits[2] << 2
    else:
        return None

def read_config():
    defaults = {"camera-dev": "-1",
                "save-filename-pattern": "exam-%%03d.png"}
    config = ConfigParser.SafeConfigParser(defaults)
    config.read([os.path.expanduser('~/.camgrade.cfg')])
    return config

def read_cmd_options():
    parser = OptionParser(usage = "usage: %prog [options]",
                          version = program_name + ' ' + version)
    parser.add_option("-e", "--exam-data-file", dest = "ex_data_filename",
                      help = "read model data from FILENAME")
    parser.add_option("-a", "--answers-file", dest = "answers_filename",
                      help = "write students' answers to FILENAME")
    parser.add_option("-s", "--start-id", dest = "start_id", type = "int",
                      help = "start at the given exam id",
                      default = 0)
    parser.add_option("-o", "--output-dir", dest = "output_dir",
                      help = "store captured images at the given directory")
    parser.add_option("-d", "--debug", action="store_true", dest = "debug",
                      default = False, help = "activate debugging features")
    parser.add_option("-c", "--camera", type="int", dest = "camera_dev",
                      help = "camera device to be selected (-1 for default)")
    parser.add_option("--stats", action="store_true", dest = "save_stats",
                      default = False,
                      help = "save performance stats to the answers file")
    parser.add_option("--id-list", dest = "ids_file", default = None,
                      help = "file with the list of valid student ids")
    parser.add_option("--capture-raw", dest = "raw_file", default = None,
                      help = "capture from raw file")
    parser.add_option("--capture-proc", dest = "proc_file", default = None,
                      help = "capture from pre-processed file")
    parser.add_option("-f", "--ajust-first", action="store_true",
                      dest = "adjust", default = False,
                      help = "don't lock on an exam until SPC is pressed")

    (options, args) = parser.parse_args()
    if options.raw_file is not None and options.proc_file is not None:
        parser.error("--capture-raw and --capture-proc are mutually exclusive")
    return options

def cell_clicked(image, point):
    min_dst = None
    clicked_row = None
    clicked_col = None
    for i, row in enumerate(image.centers):
        for j, center in enumerate(row):
            dst = imageproc.distance(point, center)
            if min_dst is None or dst < min_dst:
                min_dst = dst
                clicked_row = i
                clicked_col = j
    if min_dst <= image.diagonals[i][j] / 2:
        return (clicked_row, clicked_col + 1)
    else:
        return None

def dump_camera_buffer(camera):
    for i in range(0, 6):
        imageproc.capture(camera, False)

def show_image(image, screen):
    im = opencv.adaptors.Ipl2PIL(image)
    pg_img = pygame.image.frombuffer(im.tostring(), im.size, im.mode)
    screen.blit(pg_img, (0,0))
    pygame.display.flip()

def select_camera(options, config):
    if options.camera_dev is None:
        try:
            camera = config.getint('default', 'camera-dev')
        except:
            camera = -1
    else:
        camera = options.camera_dev
    return camera

def main():
    options = read_cmd_options()
    config = read_config()
    save_pattern = config.get('default', 'save-filename-pattern')

    if options.ex_data_filename is not None:
        solutions, dimensions, id_num_digits = \
            process_exam_data(options.ex_data_filename)
    else:
        solutions = []
        dimensions = []
        id_num_digits = 0
    read_id = (id_num_digits > 0)
    if options.output_dir is not None:
        save_pattern = os.path.join(options.output_dir, save_pattern)
    if options.answers_filename is not None:
        answers_file = options.answers_filename
    else:
        answers_file = 'camgrade-answers.txt'
        if options.output_dir is not None:
            answers_file = os.path.join(options.output_dir, answers_file)

    im_id = options.start_id
    valid_student_ids = None
    if read_id and options.ids_file is not None:
        ids_file = open(options.ids_file)
        valid_student_ids = [line.strip() for line in ids_file]
        ids_file.close()

    fps = 8.0
    pygame.init()
    window = pygame.display.set_mode((640,480))
    pygame.display.set_caption("camgrade")
    screen = pygame.display.get_surface()

    profiler = PerformanceProfiler()

    # Initialize options
    imageproc_options = imageproc.ExamCapture.get_default_options()
    imageproc_options['infobits'] = True
    if read_id:
        imageproc_options['read-id'] = True
        imageproc_options['id-num-digits'] = id_num_digits
    if options.debug:
        imageproc_options['show-status'] = True

    # Initialize capture source
    camera = None
    if options.proc_file is not None:
        imageproc_options['capture-from-file'] = True
        imageproc_options['capture-proc-file'] = options.proc_file
    elif options.raw_file is not None:
        imageproc_options['capture-from-file'] = True
        imageproc_options['capture-raw-file'] = options.raw_file
    else:
        camera = init(select_camera(options, config))

    # Program main loop
    lock_mode = not options.adjust
    while True:
        profiler.count_capture()
        image = imageproc.ExamCapture(camera, dimensions, imageproc_options)
        image.detect()
        success = image.success
        if success:
            model = decode_model_2x31(image.bits)
            if model is not None:
                exam = Exam(image, model, solutions[model], valid_student_ids,
                            im_id, options.save_stats)
                exam.grade()
                exam.draw_answers()
            else:
                success = False

        events = pygame.event.get()
        for event in events:
            if event.type == QUIT or \
                    (event.type == KEYDOWN and event.key == 27):
                sys.exit(0)
            elif event.type == KEYDOWN:
                if event.key == ord('p') and options.debug:
                    imageproc_options['show-image-proc'] = \
                        not imageproc_options['show-image-proc']
                elif event.key == ord('l') and options.debug:
                    imageproc_options['show-lines'] = \
                        not imageproc_options['show-lines']
                elif event.key == 32:
                    lock_mode = True

        show_image(image.image_drawn, screen)
        if success and lock_mode:
            continue_waiting = True
            while continue_waiting:
                event = pygame.event.wait()
                if event.type == QUIT:
                    sys.exit(0)
                elif event.type == KEYDOWN:
                    if event.key == 27:
                        sys.exit(0)
                    elif event.key == 8:
                        continue_waiting = False
                    elif event.key == 32:
                        stats = profiler.finish_exam(exam)
                        exam.save_image(save_pattern)
                        exam.save_answers(answers_file, stats)
                        if options.debug:
                            exam.save_debug_images(save_pattern)
                        im_id += 1
                        continue_waiting = False
                    elif event.key == ord('i') and read_id:
                        exam.invalidate_id()
                        show_image(exam.image.image_drawn, screen)
                    elif event.key == 9 and read_id \
                            and options.ids_file is not None:
                        if len(exam.student_id_filter) == 0:
                            exam.try_next_student_id()
                        else:
                            exam.reset_student_id_editor()
                        profiler.count_student_id_change()
                        show_image(exam.image.image_drawn, screen)
                    elif event.key >= ord('0') and event.key <= ord('9') \
                             and read_id and options.ids_file is not None:
                        exam.filter_student_id(chr(event.key))
                        profiler.count_student_id_change()
                        show_image(exam.image.image_drawn, screen)
                    elif event.key == ord('p') and options.debug:
                        imageproc_options['show-image-proc'] = \
                            not imageproc_options['show-image-proc']
                        continue_waiting = False
                    elif event.key == ord('l') and options.debug:
                        imageproc_options['show-lines'] = \
                            not imageproc_options['show-lines']
                        continue_waiting = False
                elif event.type == MOUSEBUTTONDOWN and event.button == 1:
                    cell = cell_clicked(exam.image, event.pos)
                    if cell is not None:
                        question, answer = cell
                        exam.toggle_answer(question, answer)
                        show_image(exam.image.image_drawn, screen)
            dump_camera_buffer(camera)
        else:
            pygame.time.delay(int(1000 * 1.0/fps))

if __name__ == "__main__":
    main()
