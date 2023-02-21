import numpy as np
import argparse
import sys
import math
import os
from bisect import bisect_left, bisect_right

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
import matplotlib.animation as animation

class Delta:
    def __init__(self, start, end, data, prev_data=None, is_reset=False):
        self.start = start
        self.end = end
        self.data = data.copy()
        self.prev = prev_data.copy() if prev_data is not None else None
        self.is_reset = is_reset

class ProfileSet:
    ''' Model the sequence of Gram-Schmidt vector lengths '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.__n = 0
        self.__cur_data = np.zeros(1)
        self.__deltas = []
        self.__times = []
        self.__pos = 0
        self._follow = True
        self._expect_reset = False
        self._max_time = None

    def reset_profile(self, dim):
        self._expect_reset = True

    def append(self, start, end, values, time):
        # Expand if necessary
        #assert end <= self.__n

        delta = Delta(start, end, values, is_reset=self._expect_reset)
        self._expect_reset = False

        # Is this the first delta? Apply it.
        if len(self.__deltas) == 0:
            self.__n = end
            self.__cur_data = np.zeros(end)
            self.__cur_data[start:end] = values
        self.__deltas.append(delta)
        self.__times.append(time)
        if self._follow:
            to_adv = len(self.__deltas) - self.__pos
            assert to_adv > 0
            self.advance(to_adv)

    def get_data(self):
        return self.__cur_data.copy()

    def get_time(self):
        return self.__times[self.__pos]

    def get_times(self):
        return self.__times[:]

    def _step_forward(self):
        if self.__pos >= len(self.__deltas) - 1:
            self._follow = True
            return

        delta = self.__deltas[self.__pos + 1]
        s = delta.start
        e = delta.end
        if delta.is_reset:
            if delta.prev is None:
                delta.prev = self.__cur_data.copy()
            self.__cur_data = np.array(delta.data)
        else:
            if delta.prev is None:
                delta.prev = self.__cur_data[s:e].copy()
            self.__cur_data[s:e] = delta.data
        self.__pos += 1

    def _step_backward(self):
        self._follow = False
        if self.__pos <= 0:
            return

        delta = self.__deltas[self.__pos]
        assert delta.prev is not None
        s = delta.start
        e = delta.end

        if delta.is_reset:
            self.__cur_data = delta.prev.copy()
        else:
            self.__cur_data[s:e] = delta.prev
        self.__pos -= 1

    def advance(self, steps):
        if steps < 0:
            steps = -steps
            while steps > 0:
                self._step_backward()
                steps -= 1
        else:
            while steps > 0:
                self._step_forward()
                steps -= 1

    def advance_to_time(self, t):
        # Which index is closest to the specified time?
        ts = np.array(self.__times)

        i = np.argmax(ts > t)
        self.position = i + 1

    @property
    def max_time(self):
        if self._max_time is None:
            self._max_time = np.max(self.__times)
        return self._max_time

    @property
    def position(self):
        return self.__pos + 1

    @position.setter
    def position(self, p):
        delta = p - self.position
        self.advance(delta)

    @property
    def count(self):
        return len(self.__deltas)

class AnimInds:
    def __init__(self, times, step_size):
        self.times = np.array(times)
        self.step_size = step_size
        max_time = np.max(self.times)

        count = len(self.times)
        ts = self.times
        self.arr_ind = []
        self.arr_frame = []
        self.max_frame = math.ceil(max_time / step_size)
        
        self._append(1, 0)
        
        t = step_size
        frame = 1

        ind = 0
        while ind < len(ts) - 1 and t < max_time:
            if t >= ts[ind + 1]:
                ind += 1
            else:
                self._append(ind + 1, frame)
                t += self.step_size
                frame += 1
        self._append(count, frame)

    def _append(self, ind, frame):
        if len(self.arr_ind) == 0 or self.arr_ind[-1] != ind:
            self.arr_ind += [ind]
            self.arr_frame += [frame]

    def lookup(self, frame):
        # Given a number of time steps, what is the
        # last index where the recorded time is less
        # than the elapsed time?
        ind = bisect_right(self.arr_frame, frame) - 1
        return self.arr_ind[ind]

    def reverse_lookup(self, val):
        # Given an index into the times, what
        # frame does that index fall under?
        ind = bisect_left(self.arr_ind, val)
        return self.arr_frame[ind]

    def count(self):
        # How many frames are there?
        return self.max_frame + 1
        

class MPLController:
    def __init__(self, fname, speedup=1.0, animation_file=None):
        # Set self.prof_set
        self.fname = fname
        self.parse_logfile(fname)

        self.step_size = 1
        self.plotted_data = None
        self.anim_paused = True
        self.anim_period = 0.1
        self.speedup = speedup
        self.animation_file = animation_file
        self.anim_ind = 0
        self.anim_time = 0.
        self.autoscale = True
        self.orig_ylim = (0, 1)

    def make_anim_inds(self):
        # Make lookup table for current time/animation index
        ts = np.array(self.prof_set.get_times())
        self.anim_inds = AnimInds(ts, self.anim_period * self.speedup)

    def on_press(self, evt):
        if evt.key == "right":
            self.prof_set.advance(self.step_size)
        elif evt.key == "left":
            self.prof_set.advance(-self.step_size)
        elif evt.key == "up":
            self.step_size *= 2
        elif evt.key == "down":
            self.step_size //= 2
            self.step_size = max(1, self.step_size)
        elif evt.key == " ":
            self.on_play_clicked(None)
        else:
            #print(f"Keypress {evt.key}")
            pass
        self.freq_slider.set_val(self.prof_set.position)

    def on_slider_changed(self, val):
        self.prof_set.position = val
        self.anim_ind = self.anim_inds.reverse_lookup(int(val))
        self.anim_time = self.anim_ind * self.anim_period * self.speedup
        self.update()

    def on_play_clicked(self, val):
        self.anim_paused = not self.anim_paused
        self.update()

    def on_autoscale_clicked(self, val):
        self.autoscale = not self.autoscale
        self.update()

    def on_anim_params_update(self, val):
        try:
            speed = float(val)
            if speed <= 0.:
                raise ValueError
        except ValueError:
            speed = 1.0
        self.speedup = speed
        self.anim_ind = 0
        self.anim_time = self.anim_ind * self.anim_period * self.speedup
        self.make_anim_inds()

    def update_anim(self, ind):
        if ind == 0:
            self.anim_ind = 0
            self.anim_time = 0
        self.anim_ind %= self.anim_inds.count()
        ind = self.anim_inds.lookup(self.anim_ind)
        #self.prof_set.advance_to_time(self.anim_time)
        self.prof_set.position = ind
        new_anim_ind = self.anim_ind + 1
        self.freq_slider.set_val(self.prof_set.position)
        self.anim_ind = new_anim_ind
        self.anim_time = self.anim_ind * self.anim_period * self.speedup
        self.update()

    def update(self):
        if self.anim.event_source:
            if self.anim_paused:
                self.anim.event_source.stop()
            else:
                self.anim.event_source.start()

        self.btn_play.label.set_text("Play" if self.anim_paused else "Pause")

        self.btn_autoscale.label.set_text(
            "Disable Autoscale" if self.autoscale else
            "Enable Autoscale"
        )
            
        gsvs = self.prof_set
        data = gsvs.get_data()
        t = gsvs.get_time()
        step = gsvs.position
        count = gsvs.count

        if self.plotted_data is None:
            if np.any(np.isnan(data)):
                nan_inds = np.isnan(data)
                data[nan_inds] = 0
                self.plotted_data, = self.ax.plot(data)
                data[nan_inds] = float('nan')
                self.plotted_data.set_ydata(data)
            else:
                self.plotted_data, = self.ax.plot(data)
            self.orig_ylim = self.ax.get_ylim()
            self.num_xs = data.shape[0]
        else:
            if data.shape[0] != self.num_xs:
                self.ax.ignore_existing_data_limits = True
                self.num_xs = data.shape[0]
                if np.any(np.isnan(data)):
                    nan_inds = np.isnan(data)
                    data[nan_inds] = 0
                    self.plotted_data.set_data(range(self.num_xs), data)
                    if self.autoscale:
                        self.ax.relim()
                    data[nan_inds] = float('nan')
                    self.plotted_data.set_ydata(data)
                else:
                    self.plotted_data.set_data(range(self.num_xs), data)
                    if self.autoscale:
                        self.ax.relim()
                if self.autoscale:
                    self.ax.autoscale_view(True,True,True)
                    self.orig_ylim = self.ax.get_ylim()
            else:
                nan_inds = np.isnan(data)
                if np.any(nan_inds):
                    data[nan_inds] = 0
                self.plotted_data.set_ydata(data)
                if self.autoscale:
                    self.ax.relim()
                if np.any(nan_inds):
                    data[nan_inds] = float("nan")
                    self.plotted_data.set_ydata(data)
                if self.autoscale:
                    self.ax.autoscale_view(True,True,True)
        if not self.autoscale:
            self.ax.set_ylim(self.orig_ylim, auto=None)

        self.ax.set_title(f"Profile at {self.anim_time:0.01f}s ({step}/{count})")
        self.ax.set_ylabel("log_2 | b^*_i|")
        self.fig.canvas.draw_idle()

    def run(self):
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.15, bottom=0.2)

        # Make a horizontal slider to control the frequency.
        axfreq = self.fig.add_axes([0.15, 0.1, 0.75, 0.03])
        self.freq_slider = Slider(
            ax=axfreq,
            label='Iteration',
            valmin=1,
            valmax=self.prof_set.count,
            valinit=self.prof_set.count,
            valstep=1,
        )
        self.freq_slider.on_changed(self.on_slider_changed)

        axplay = self.fig.add_axes([0.15, 0.03, 0.1, 0.06])
        self.btn_play = Button(axplay, 'Play')
        self.btn_play.on_clicked(self.on_play_clicked)

        axspeed = self.fig.add_axes([0.46, 0.03, 0.1, 0.06])
        self.txt_speed = TextBox(axspeed, "Speedup ", initial=str(self.speedup))
        self.txt_speed.on_submit(self.on_anim_params_update)

        axscale = self.fig.add_axes([0.68, 0.03, 0.22, 0.06])
        self.btn_autoscale = Button(axscale, "Disable Autoscale")
        self.btn_autoscale.on_clicked(self.on_autoscale_clicked)

        self.make_anim_inds()

        if self.animation_file is not None:
            frames = self.anim_inds.count()
        else:
            frames = None
            
        self.anim = animation.FuncAnimation(
            self.fig, self.update_anim, frames=frames, fargs=None, interval=int(1000 * self.anim_period))

        if self.animation_file is not None:
            # save animation at 30 frames per second
            self.anim_paused = False
            #fps = int(1/self.anim_period)
            self.anim.save(self.animation_file) #, writer='imagemagick', fps=30)
            return

        self.fig.canvas.mpl_connect('key_press_event', self.on_press)
        self.fig.canvas.manager.set_window_title(f"flatter profile: {self.fname}")

        self.update()
        plt.show()

    def parse_logfile(self, fname):
        self.prof_set = ProfileSet()
        with open(fname) as f:
            for line in f:
                if "profile" in line:
                    self.parse_profile_line(line)
        self.prof_set.position = 1

    def parse_profile_line(self, ln):
        prof_set = self.prof_set
        ln = ln.rstrip()
        end_i = ln.find(")")
        start_i = ln.find("profile")
        if "," not in ln[start_i + 8:end_i]:
            # This is a reset line
            dim = int(ln[start_i+8:end_i])
            prof_set.reset_profile(dim)
        else:
            bounds = ln[start_i+8:end_i].split(",")
            start = int(bounds[0])
            end = int(bounds[1])

            ts = ln[ln.find("[")+1:ln.find("]")]
            ts = float(ts)

            end_i = ln.find("]")

            val_pairs = [c.split("+") for c in ln[end_i+1:].split(" ") if len(c) > 0]

            if len(val_pairs) != end - start:
                raise ValueError("Unexpected number of values for profile line")

            vals = [float(v[0])+float(v[1]) for v in val_pairs]

            prof_set.append(start, end, vals, ts)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile", nargs='?', type=str, help="flatter log (default FLATTER_LOG)")
    parser.add_argument("--speedup", type=float, default=1.0, help="Speedup factor for replay")
    parser.add_argument("--animation", type=str, help="Save animation")

    args = parser.parse_args()

    if args.logfile is not None:
        fname = args.logfile
    elif "FLATTER_LOG" in os.environ:
        fname = os.environ["FLATTER_LOG"]
    else:
        print(f"Could not open logfile")
        sys.exit(-1)
    
    speedup = args.speedup
    anim = args.animation
    con = MPLController(fname, speedup=speedup, animation_file=anim)
    con.run()

if __name__ == "__main__":
    main()
