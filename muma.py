"""
traffic_sim.py  Version 2.0


OVERVIEW

This Python script simulates automotive traffic arriving at a traffic light.
The script demonstrates the following:

(1) several basic principles of discrete-event simulation, including the
representation of system states, the connection between events and changes of
state, and the Poisson process,

(2) the use of the SimPy module,

(3) simulation of random deviates using NumPy, and

(4) the use of Python generator functions.


THE MODEL

Cars drive on a single-lane road and arrive at the intersection from one
direction only according to a Poisson process with specified rate.  Positions
and velocities of the cars are not modeled, and there is thus no coordinate
system.

The traffic light switches between green and red at regular intervals; there is
no yellow state.

When a car arrives at a green light with no cars queued, it passes immediately
through the intersection and departs the simulation.

When a car arrives at a red light or with one or more cars queued at the
intersection, it stops and joins the queue.

When the light turns green, if there is a queue, cars enter the intersection one
at a time.  Once a car has entered the intersection, it clears the intersection,
regardless of the state of the light, after a delay that is a random draw from a
triangular distribution.


INPUTS

All inputs are hardwired into the code.


OUTPUTS

The simulation generates a trace of events as it runs, and also reports
estimates of the average number of cars in the queue and the average waiting
time.


FUTURE WORK

This model might be extended in any of several directions.  Some possibilities
include the following:

(1) There are several sources of bias in the average waiting time: (a) The
simulation starts with a green light.  (b) The simulation starts with no cars
queued.  (c) Cars remaining in the queue at the end of the simulation do not
contribute to the estimated waiting time.  The effects of (a) and (b) might be
mitigated by eliminating the initial transient; this would involve collecting
and saving individual waiting times, choosing alternative values for the
duration of the initial transient, generating estimates using only the data
outside of the transient interval, and then selecting one of these results using
some statistical criteria. (We want the mean of the selected result to have no
statistically significant difference from results based on longer transient
intervals, but otherwise we want to discard as little of the data as possible).

(2) The delay of the driver at the head of the queue in responding to the change
of the light from red to green could be made somewhat longer than that of
drivers who are further back in the queue.  This would allow for greater realism
in the model without explicitly modeling positions and velocities of the cars.

(3) One might have some fraction of the cars make a right-turn at the light,
with the remaining cars going straight, and compare the capacity of a
single-lane road, a two-lane road with no turn restrictions, and a two-lane road
with a dedicated right-turn-only lane.


AUTHOR

Dr. Phillip M. Feldman
"""


# Section 1: Import from modules and define a utility class.

from collections import deque # double-ended queue
from numpy import random
import simpy, itertools, operator
from simpy.util import start_delayed
from sys import argv


class Struct(object):
   """
   This simple class allows one to create an object whose attributes are
   initialized via keyword argument/value pairs.  One can update the attributes
   as needed later.
   """
   def __init__(self, **kwargs):
      self.__dict__.update(kwargs)


# Section 2: Initializations.

random.seed([1, 2, 3])

# Cars cars arrive at the traffic light according to a Poisson process with an
# average rate of 0.2 per second:
arrival_rate= 0.2
t_interarrival_mean= 1.0 / arrival_rate
t_green= 5.0; t_red= 5.0


# The time for a car at the head of the queue to depart (clear the intersection)
# is modeled as a triangular distribution with specified minimum, maximum, and
# mode.
t_depart_left= 1.6; t_depart_mode= 2.0; t_depart_right= 2.4

# Initially, no cars are waiting at the light:
q1 = deque()
q2 = deque()
q3 = deque()
q4 = deque()

lanes = [q1,q2,q3,q4]
lanes_to_leave = ["q11","q21","q31","q41"]

collision_oracle = [
   "q1q21+q2q41",
   "q1q21+q2q31",
   "q1q21+q3q11",
   "q1q21+q3q41",
   "q1q21+q4q11",
   "q1q31+q2q41",
   "q1q31+q3q41",
   "q1q31+q4q11",
   "q1q31+q4q21",

   "q2q31+q1q21",
   "q2q31+q3q11",
   "q2q31+q3q41",
   "q2q31+q4q21",
   "q2q31+q4q11",
   "q2q41+q4q11",
   "q2q41+q1q31",
   "q2q41+q1q21",
   "q2q41+q3q11",

   "q3q11+q1q21",
   "q3q11+q2q31",
   "q3q11+q2q41",
   "q3q11+q4q21",
   "q3q41+q4q11",
   "q3q41+q4q21",
   "q3q41+q1q31",
   "q3q41+q1q21",
   "q3q41+q2q31",

   "q4q11+q1q21",
   "q4q11+q1q31",
   "q4q11+q2q41",
   "q4q11+q2q31",
   "q4q11+q3q41",
   "q4q21+q3q11",
   "q4q21+q3q41",
   "q4q21+q1q31",
   "q4q21+q2q31"
]

# Track number of cars:
arrival_count= departure_count= 0

Q_stats= Struct(count=0, cars_waiting=0)
W_stats= Struct(count=0, waiting_time=0.0)


# Section 3: Arrival event.

def arrival():
   """
   This generator functions simulates the arrival of a car.  Cars arrive
   according to a Poisson process having rate `arrival_rate`.  The times between
   subsequent arrivals are i.i.d. exponential random variables with mean

      t_interarrival_mean= 1.0 / arrival_rate
   """
   global arrival_count, departure_count, env, q1, q2, q3, q4, lanes_to_leave

   while True:
      arrival_count+= 1
      current_lane_number = random.randint(0, 4)
      random_lane = lanes[current_lane_number]
      if current_lane_number == "q1":
         leaving_lane = random.choice(["q21", "q31", "q41"])
      elif current_lane_number == "q2":
         leaving_lane = random.choice(["q11", "q31", "q41"])
      elif current_lane_number == "q3":
         leaving_lane = random.choice(["q21", "q11", "q41"])
      else:
         leaving_lane = random.choice(["q21", "q31", "q11"])

      # if len(random_lane):

      # There is a queue of cars.  ==> The new car joins
      # the queue.  Append a tuple that contains the number of the car and
      # the time at which it arrived:
      current_lane = "q"+str(current_lane_number+1)
      budget = random.randint(0, 100)
      random_lane.append((arrival_count, current_lane, leaving_lane, current_lane+leaving_lane, budget, env.now))
      print("Car #%d arrived and joined the lane %s and leaving towards %s at position %d at time "
        "%.3f." % (arrival_count, current_lane, leaving_lane, len(current_lane), env.now))

      # else:

      #    # The light is green and no cars are waiting.  ==> The new car passes
      #    # through the intersection immediately.
      #    print("Car #%d arrived to a green light with no cars waiting at time "
      #      "%.3f." % (arrival_count, env.now))

      #    # Record waiting time statistics.  (This car experienced zero waiting
      #    # time, so we increment the count of cars, but the cumulative waiting
      #    # time remains unchanged.
      #    W_stats.count+= 1

      # Schedule next arrival:
      yield env.timeout( random.exponential(t_interarrival_mean))


def auction(items, details):
   winners = []
   for lanes in items:
      budgets = {}
      for lane in lanes:
         budgets[lane] = (filter(lambda x: x[2] == lane, details))[0][4]
      winners.append(max(budget.iteritems(), key=operator.itemgetter(1))[0])

   return winners

def run_auction(items, lane_map):
   global env, q1, q2, q3, q4

   routes = map(lambda car: car[3], items)
   route_combos = filter(lambda x: x[0] != x[1], itertools.product(routes, routes))
   routes_taken = map(lambda x: "+".join([x[0], x[1]]), route_combos)

   clashing_routes = filter(lambda route_combo: route_combo in collision_oracle, routes_taken)
   lanes_in_auction = []
   winners = []

   for lane_combo in clashing_routes:
      lane1, lane2 = map(lambda x: x[0:2], lane_combo.split("+"))

      for lane_set in lanes_in_auction:
         if lane1 in lane_set or lane2 in lane_set:
            lane_set.add(lane1)
            lane_set.add(lane2)
            break

   if len(lanes_in_auction) > 0 and len(lanes_in_auction[0].intersection(lanes_in_auction[1])) != 0:
      lanes_in_auction = [lanes_in_auction[0].union(lanes_in_auction[1])]
   
   winners = auction(lanes_in_auction, items)

   for lane_combo in clashing_routes:
      routes_taken.remove(lane_combo)
   
   for lane_combo in routes_taken:
      lane1, lane2 = map(lambda x: x[0:2], lane_combo.split("+"))
      if lane1 not in lanes_in_auction:
         winners.append(lane1)
      if lane2 not in lanes_in_auction:
         winners.append(lane2)

   return list(set(winners))

# Section 4: Define event functions.

# Section 4.1: Departure event.

def departure():
   """
   This generator function simulates the 'departure' of a car, i.e., a car that
   previously entered the intersection clears the intersection.  Once a car has
   departed, we remove it from the queue, and we no longer track it in the
   simulation.
   """
   global env, departure_count, q1, q2, q3, q4

   lane_map = {"q1":q1, "q2":q2, "q3":q3, "q4":q4}
   car_map = {}

   while True:

      # The car that entered the intersection clears the intersection:

      cars = []
      if len(q1) > 0:
         car1 = q1.popleft()
         car_map["q1"] = car1
         cars.append(car1)
      if len(q2) > 0:
         car2 = q2.popleft()
         car_map["q2"] = car2
         cars.append(car2)
      if len(q3) > 0:
         car3 = q3.popleft()
         car_map["q3"] = car3
         cars.append(car3)
      if len(q4) > 0:
         car4 = q4.popleft()
         car_map["q4"] = car4
         cars.append(car4)

      winners = run_auction(cars, lane_map)

      losers = filter(lambda x: x not in winners, car_map.keys())
      # print len(q1)
      # print len(q2)
      # print len(q3)
      # print len(q4)
      # print winners
      # print losers
      # print car_map
      # print "*"*100
      for loser in losers:
         lane_map[loser].appendleft(car_map[loser])

      for winner in winners:
         car_number, lane, left_in, combo, budget, t_arrival= car_map[winner]
         print("Car #%d departed lane %s at time %.3f, leaving %d cars in the queue."
           % (car_number, lane, env.now, len(lane_map[lane])))

         departure_count += 1
         # Record waiting time statistics:
         W_stats.count+= 1
         W_stats.waiting_time+= env.now - t_arrival

      # If the light is red or the queue is empty, do not schedule the next
      # departure.  `departure` is a generator, so the `return` statement
      # terminates the iterator that the generator produces.
      if len(q4) == 0 or len(q3) == 0 or len(q2) == 0 or len(q1) == 0:
         return

      # Generate departure delay as a random draw from triangular distribution:
      delay= random.triangular(left=t_depart_left, mode=t_depart_mode,
        right=t_depart_right)

      # Schedule next departure:
      yield env.timeout(delay)



def start():
   global env

   while True:

      # If there are cars in the queue, schedule a departure event:
      if len(q1) or len(q2) or len(q3) or len(q4):

         # Generate departure delay as a random draw from triangular
         # distribution:
         delay= random.triangular(left=t_depart_left, mode=t_depart_mode,
           right=t_depart_right)

         start_delayed(env, departure(), delay=delay)

      yield env.timeout(t_green)
      yield env.timeout(t_red)


# Section 4.3: Schedule event that collects Q_stats.

def monitor():
   """
   This generator function produces an interator that collects statistics on the
   state of the queue at regular intervals.  An alternative approach would be to
   apply the PASTA property of the Poisson process ('Poisson Arrivals See Time
   Averages') and sample the queue at instants immediately prior to arrivals.
   """
   global env, Q_stats

   while True:
      Q_stats.count+= 1
      Q_stats.cars_waiting+= len(q1) + len(q2) + len(q3) + len(q4)
      yield env.timeout(1.0)


# Section 5: Schedule initial events and run the simulation.  Note: The first
# change of the traffic light, first arrival of a car, and first statistical
# monitoring event are scheduled by invoking `env.process`.  Subsequent changes
# will be scheduled by invoking the `timeout` method.  With this scheme, there
# is only one event of each of these types scheduled at any time; this keeps the
# event queue short, which is good for both memory utilization and running time.
file_csv = open("mma.txt", "a+")

print("\nSimulation of Cars Arriving at Intersection Controlled by a Traffic\n\n")
i = argv[1]
# Total number of seconds to be simulated:
end_time= float(i)

# Initialize environment:
env= simpy.Environment()

# Schedule first change of the traffic light:
env.process(start())

# Schedule first arrival of a car:
t_first_arrival= random.exponential(t_interarrival_mean)
start_delayed(env, arrival(), delay=t_first_arrival)

# Schedule first statistical monitoring event:
env.process(monitor())

# Let the simulation run for specified time:
env.run(until=end_time)


# Section 6: Report statistics.

print("\n\n      *** Statistics ***\n\n")

print("Mean number of cars waiting: %.3f"
  % (Q_stats.cars_waiting / float(Q_stats.count)))

print("Number of cars arrived: %.3f"
  % arrival_count)

print("Number of cars departed: %.3f"
  % departure_count)

print("Mean waiting time (seconds): %.3f"
  % (W_stats.waiting_time / float(W_stats.count)))

string = "\n"

string += str(i)+"\t"
string += "{:.2f}".format(Q_stats.cars_waiting / float(Q_stats.count))+"\t"
string += "{:.2f}".format((departure_count*1.0/arrival_count)*100)+"\t"
string += "{:.2f}".format(W_stats.waiting_time / float(W_stats.count))+"\t"

string += str(arrival_count)+"\t"
string += str(departure_count)+"\t"
string += str(len(q1)+len(q2)+len(q3)+len(q4))+"\t"

file_csv.write(string)

file_csv.close()
