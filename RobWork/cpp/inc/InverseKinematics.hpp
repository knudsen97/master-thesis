#ifndef __INVERSEKINEMATICS_H__
#define __INVERSEKINEMATICS_H__
// Standard library
#include <iostream>

// RobWork
#include <rw/core/Ptr.hpp>
#include <rw/kinematics/State.hpp>
#include <rw/kinematics/Kinematics.hpp>
#include <rw/loaders/WorkCellLoader.hpp>
#include <rw/loaders/path/PathLoader.hpp>

#include <rw/math/RPY.hpp>
#include <rw/pathplanning.hpp>
#include <rw/math/Rotation3D.hpp>
#include <rw/invkin.hpp>
#include <rw/models/SerialDevice.hpp>
#include <rw/proximity/CollisionDetector.hpp>
#include <rw/proximity/CollisionStrategy.hpp>

#include <rwlibs/proximitystrategies/ProximityStrategyYaobi.hpp>


class InverseKinematics
{
private:
    static constexpr double invalid = -99;
public:
    /**
     * @brief Construct a new Inverse Kinematics object.
     * @param robot The robot to solve inverse kinematics for.
     * @param wc The workcell to solve inverse kinematics in.
    */
    InverseKinematics(rw::models::SerialDevice::Ptr robot, rw::core::Ptr<rw::models::WorkCell> wc);
    ~InverseKinematics();

public:
    /**
     * @brief Solve inverse kinematics for a given target. The data is available through the getters.
     * @param WorldTTarget The target transform in world coordinates.
     * @param rollStart The roll angle to solve from in radians. Default = 0.
     * @param rollEnd The roll angle to solve to in radians. Default = -99 or rollStart if rollStart is given.
     * @param step The step size for the roll rotation. Default = 1.
     * @return True if a solution was found, false otherwise.
    */
    bool solve(rw::math::Transform3D<> WorldTTarget, double rollStart = 0, double rollEnd = invalid, double step = 1);

    /**
     * @brief Get the solutions.
     * @return A vector of solutions.
    */
    std::vector<rw::math::Q> getSolutions();

    /**
     * @brief Get the solutions.
     * @param solutions A vector of solutions.
    */
    void getSolutions(std::vector<rw::math::Q>& solutions);

    /**
     * @brief Get the collision free solutions replay.
     * @return A vector of collision free solutions.
    */
    rw::trajectory::TimedStatePath getReplay();

    /**
     * @brief Get the collision free solutions replay.
     * @param replay The variable the .
    */
    void getReplay(rw::trajectory::TimedStatePath& replay);

private:
    rw::core::Ptr<rw::models::WorkCell> workCell_;
    rw::kinematics::State state_;
    rw::models::SerialDevice::Ptr robot_;
    rw::invkin::ClosedFormIKSolverUR::Ptr solver_;
    rw::proximity::CollisionStrategy::Ptr collisionStrategy_;
    rw::proximity::CollisionDetector::Ptr collisionDetector_;
    std::vector<rw::math::Q> collisionFreeSolutions_;
    rw::trajectory::TimedStatePath collisionFreeStates_;
};


#endif // __INVERSEKINEMATICS_H__