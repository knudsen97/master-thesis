#ifndef __MOTIONPLANNING_H__
#define __MOTIONPLANNING_H__


// Standard library
#include <iostream>

// RobWork
#include <rw/core/Ptr.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
#include <rw/loaders/path/PathLoader.hpp>

#include <rw/math/RPY.hpp>
#include <rw/math/Rotation3D.hpp>
#include <rw/math/MetricFactory.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/kinematics/Kinematics.hpp>
#include <rw/invkin.hpp>

#include <rw/models/SerialDevice.hpp>
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/proximity/CollisionStrategy.hpp>
#include <rwlibs/proximitystrategies/ProximityStrategyYaobi.hpp>

#include <rw/pathplanning.hpp>
#include <rwlibs/pathplanners/rrt/RRTPlanner.hpp>
#include <rw/pathplanning/PlannerConstraint.hpp>

#include <rw/trajectory.hpp>
#include "../inc/InverseKinematics.hpp"



class MotionPlanning
{
public:
    /**
     * @brief Construct a new Motion Planning object.
     * @param robot The robot to solve inverse kinematics for.
     * @param wc The workcell to solve inverse kinematics in.
     * @param state The state to solve inverse kinematics in.
    */
    MotionPlanning(rw::models::SerialDevice::Ptr robot, rw::core::Ptr<rw::models::WorkCell> wc,  rw::kinematics::State state);
    ~MotionPlanning();

public:

    /**
     * @brief Sets the distance to the subgoal. This introduces a subgoal. Default is go directly to the target.
     * @param distance The distance (in meters) to the subgoal.
    */
    void setSubgoalDistance(double distance);

    /**
     * @brief Sets the transform from world to object.
     * @param worldTobject The transform from world to object.
    */
    void setWorldTobject(rw::math::Transform3D<> worldTobject);

    /**
     * @brief Sets the target transform.
     * @param WorldTTarget The target transform in world coordinates.
    */
    void setTarget(rw::math::Q targetQ);

    /**
     * @brief Sets the start configuration.
     * @param startQ The start configuration.
    */
    void setStartQ(rw::math::Q startQ);

    /**
     * @brief Get the shortest solution from Q to vector of Qs.
     * @param Q The configuration to compare to.
     * @param QVec The vector of configurations to compare with.
    */
    rw::math::Q getShortestSolutionFromQToQVec(rw::math::Q Q, std::vector<rw::math::Q> QVec);

    /**
     * @brief Linear interpolation between two configurations.
     * @param subgoals The subgoals to interpolate between.
     * @param timestep The timestep between each subgoal.
    */
    rw::trajectory::QPath linearInterpolator(rw::trajectory::QPath subgoals, double timestep);    

    /**
     * @brief Calculate the subgoals.
     * @return A vector of subgoals.
    */
    rw::trajectory::QPath calculateSubgoals();

    /**
     * @brief Returns the linear interpolation (in Q space) between the subgoals. Called after calculateSubgoals().
     * @param timestep The timestep between each subgoal. Default is 0.1.
    */
    rw::trajectory::QPath getLinearPath(double timestep = 0.1);


    /**
     * @brief Get the collision free solutions replay.
     * @param dt The timestep between each solution. Default is 0.1.
     * @return A vector of collision free solutions.
    */
    rw::trajectory::TimedStatePath getReplay(double dt = 0.1);

    /**
     * @brief Get the collision free solutions replay.
     * @param replay Reference to store the replay in.
     * @param dt The timestep between each solution. Default is 0.1.
    */
    void getReplay(rw::trajectory::TimedStatePath& replay, double dt = 0.1);

private:
    rw::core::Ptr<rw::models::WorkCell> workCell_;
    rw::kinematics::State state_;
    rw::models::SerialDevice::Ptr robot_;
    rw::math::Q Qtarget_;
    rw::math::Q Qstart;
    rw::math::Transform3D<> worldTobject_;
    double subgoalDistance_ = 0;

    rw::trajectory::QPath subgoals_;
    std::vector<rw::math::Q> path_;
    rw::trajectory::TimedStatePath pathReplay_;

    rw::invkin::ClosedFormIKSolverUR::Ptr solver_;
    rw::proximity::CollisionStrategy::Ptr collisionStrategy_;
    rw::proximity::CollisionDetector::Ptr collisionDetector_;
    rw::math::QMetric::Ptr distanceMetric_;
    rw::pathplanning::QConstraint::Ptr Qconstraint_;
    rw::pathplanning::PlannerConstraint plannerConstraint_;
    rw::pathplanning::QSampler::QSampler::Ptr sampler_;
    rw::pathplanning::QEdgeConstraint::Ptr edgeContrain_;
    rw::pathplanning::QToQPlanner::Ptr planner_;

};
#endif // __MOTIONPLANNING_H__